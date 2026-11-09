; =============================================================================
; sa_avx2.asm -- AVX2 kernels for swap-move simulated annealing
;
; Windows x64 only. Assembler: NASM 2.15 or newer (tested with NASM 3.02)
;
; Fundamentally, the algorithm is the one in src/sa_scalar.c, step for step.
; The float operations are the same ones in the same order, so
; this code should exactly make the decision the scalar code makes for the same seed.
; We show that while the scalar code does one thing at a time, 
; AVX2 does eight, one per 32-bit lane of a 256-bit ymm register:
;
;   scalar sa_scalar.c                       here
;   ------------------------------------     ---------------------------------------------
;   g_update: g[v] += P_j[v] - P_i[v]        G_UPDATE: 8 drugs per vsubps/vaddps,
;     one drug per iteration                   4 ymm per iteration (32 drugs)
;
;   compute_g: one drug, one team member     refresh_core: 32 drugs per pass over the team
;
;   draw_random: one Philox block per step   philox8_core: 8 blocks (8 steps) per call,
;                                              structure-of-arrays, vpmuludq for the
;                                              32x32->64 multiplies
;
;   random_below: one 64-bit multiply        MULHI_U32: 8 high halves at once
;
;   neg_ln_u[r >> 20] * T: one load          vpsrld + vgatherdps + vmulps, 8 at once
;
;   propose_swap + should_accept: one step   speculative pass: 8 proposals at once with
;                                              gathers (order -> i, j; g[i], g[j];
;                                              lam_p[i][j]), 8 compares, vmovmskps +
;                                              tzcnt to jump to the first accepted one
;
;   memcpy of the best team                  COPY_TEAM: vmovdqu + vpmaskmovd tail
;
;
;

bits 64
default rel

%ifnidn __OUTPUT_FORMAT__, win64
    %error "sa_avx2.asm is Windows x64 only"
%endif

; Byte offsets of the SaAvx2Chain fields.
%define C_LAMP      0       ; const float *lam_p      n rows x stride, aligned
%define C_W         8       ; const float *w          stride, aligned
%define C_TEMPS     16      ; const float *temps      one per sweep
%define C_LNU       24      ; const float *neg_ln_u   4096 entries
%define C_TRACE     32      ; uint8_t *trace          this chain's row, or NULL
%define C_G         40      ; float *g                stride, aligned
%define C_ORDER     48      ; int32_t *order          n slots
%define C_BEST      56      ; int32_t *best_team      k drugs
%define C_STRIDE    64      ; int64_t stride          multiple of 8
%define C_TSTEPS    72      ; int64_t trace_steps
%define C_SEED      80      ; uint64_t seed
%define C_ACCEPTS   88      ; int64_t accepts         out
%define C_N         96      ; int32_t n
%define C_K         100     ; int32_t k
%define C_SWEEPS    104     ; int32_t sweeps
%define C_CHAIN     108     ; uint32_t chain_id
%define C_ENERGY    112     ; float energy            in/out
%define C_BESTE     116     ; float best_energy       in/out

REFRESH_EVERY_ACCEPTS equ 1000   ; same as sa_scalar.c
ACCEPT_TABLE_SHIFT    equ 20     ; 32 - 12 bits of L-table index

; vcmpps predicates since a NaN compares false in all cases, we can use the same predicate for < and <=
CMP_LT_OQ equ 0x11
CMP_LE_OQ equ 0x12

section .rdata rdata align=32

; RNG Generator constants for Philox 4x32-10, see src/philox.c
align 32
LANE_INDEX: dd 0, 1, 2, 3, 4, 5, 6, 7
SIGN_BIT:   times 8 dd 0x80000000        ; float sign / unsigned-compare bias
PHILOX_M0:  times 8 dd 0xD2511F53        ; Philox4x32 multipliers
PHILOX_M1:  times 8 dd 0xCD9E8D57
PHILOX_W0:  times 8 dd 0x9E3779B9        ; Philox4x32 key increments (Golden-Ratio Weyl)
PHILOX_W1:  times 8 dd 0xBB67AE85
TAIL_MASK:  times 8 dd -1                ; load 8 dwords from TAIL_MASK + 32 - 4*r to get a mask of r leading ones
            times 8 dd 0                 

; saves / restores the low 128 bits of xmm6..xmm15
%macro SAVE_XMM6_15 1
    vmovdqa [%1 + 0*16], xmm6
    vmovdqa [%1 + 1*16], xmm7
    vmovdqa [%1 + 2*16], xmm8
    vmovdqa [%1 + 3*16], xmm9
    vmovdqa [%1 + 4*16], xmm10
    vmovdqa [%1 + 5*16], xmm11
    vmovdqa [%1 + 6*16], xmm12
    vmovdqa [%1 + 7*16], xmm13
    vmovdqa [%1 + 8*16], xmm14
    vmovdqa [%1 + 9*16], xmm15
%endmacro

%macro RESTORE_XMM6_15 1
    vmovdqa xmm6,  [%1 + 0*16]
    vmovdqa xmm7,  [%1 + 1*16]
    vmovdqa xmm8,  [%1 + 2*16]
    vmovdqa xmm9,  [%1 + 3*16]
    vmovdqa xmm10, [%1 + 4*16]
    vmovdqa xmm11, [%1 + 5*16]
    vmovdqa xmm12, [%1 + 6*16]
    vmovdqa xmm13, [%1 + 7*16]
    vmovdqa xmm14, [%1 + 8*16]
    vmovdqa xmm15, [%1 + 9*16]
%endmacro

; MULHI_U32 dst, src, mul, tmp
; dst[l] = (src[l] * mul[l]) >> 32 for the 8 unsigned 32-bit lanes: the vector form of random_below(). 
; since vpmuludq multiplies only the even dwords, the odd lanes are shifted down to the even lanes, multiplied, and shifted back up to the odd lanes.

%macro MULHI_U32 4
    vpsrlq      %4, %2, 32              ; odd lanes moved to even positions
    vpmuludq    %1, %2, %3              ; 64-bit products of the even lanes
    vpmuludq    %4, %4, %3              ; 64-bit products of the odd lanes
    vpsrlq      %1, %1, 32              ; even lanes: high half down into place
    vpblendd    %1, %1, %4, 0xAA        ; odd lanes: high half is already in place
%endmacro

; One Philox4x32 round on 8 blocks
%macro PHILOX_ROUND 0
    vpsrlq      ymm6, ymm0, 32
    vpmuludq    ymm0, ymm0, [PHILOX_M0]     ; p0, even lanes
    vpmuludq    ymm6, ymm6, [PHILOX_M0]     ; p0, odd lanes
    vpsrlq      ymm7, ymm0, 32
    vpblendd    ymm7, ymm7, ymm6, 0xAA      ; hi(p0)
    vpsllq      ymm6, ymm6, 32
    vpblendd    ymm0, ymm0, ymm6, 0xAA      ; lo(p0)
    vpxor       ymm7, ymm7, ymm3
    vpxor       ymm8, ymm7, ymm5            ; c2' = hi(p0) ^ c3 ^ k1
    vmovdqa     ymm3, ymm0                  ; c3' = lo(p0)

    vpsrlq      ymm6, ymm2, 32
    vpmuludq    ymm2, ymm2, [PHILOX_M1]     ; p1, even lanes
    vpmuludq    ymm6, ymm6, [PHILOX_M1]     ; p1, odd lanes
    vpsrlq      ymm7, ymm2, 32
    vpblendd    ymm7, ymm7, ymm6, 0xAA      ; hi(p1)
    vpsllq      ymm6, ymm6, 32
    vpblendd    ymm2, ymm2, ymm6, 0xAA      ; lo(p1)
    vpxor       ymm7, ymm7, ymm1
    vpxor       ymm0, ymm7, ymm4            ; c0' = hi(p1) ^ c1 ^ k0
    vmovdqa     ymm1, ymm2                  ; c1' = lo(p1)
    vmovdqa     ymm2, ymm8                  ; c2'
%endmacro

; BUILD_COUNTERS first, scratch
; initializes the 8 Philox counters for 8 steps, and the 8 keys for 8 steps
; low 32-bit generator counter is the step number, high 32-bit counter is the chain_id, key is the seed
%macro BUILD_COUNTERS 2
    vmovq        xmm1, %1
    vpbroadcastd ymm1, xmm1                 ; lo(first) in every lane
    vpaddd       ymm0, ymm1, [LANE_INDEX]   ; lo(first) + lane, mod 2^32
    vpxor        ymm1, ymm1, [SIGN_BIT]     ; unsigned compare via signed compare
    vpxor        ymm2, ymm0, [SIGN_BIT]     ; on biased values
    vpcmpgtd     ymm2, ymm1, ymm2           ; -1 in lanes that wrapped past 2^32
    mov          %2, %1
    shr          %2, 32
    vmovq        xmm1, %2
    vpbroadcastd ymm1, xmm1                 ; hi(first)
    vpsubd       ymm1, ymm1, ymm2           ; + carry
%endmacro

; G_UPDATE g, row_j, row_i, bytes, off, end (all 64-bit registers)
; g[v] += row_j[v] - row_i[v] 
; over `bytes` bytes (a multiple of 32; all three arrays 32-byte aligned).
; function does what the C code does, but with 8 floats at a time, and with the offsets in bytes.
%macro G_UPDATE 6
    mov         %6, %4
    and         %6, -128                    ; end of the 128-byte blocks
    xor         %5, %5
    test        %6, %6
    jz          %%tail
%%block:
    vmovaps     ymm0, [%2 + %5]
    vmovaps     ymm1, [%2 + %5 + 32]
    vmovaps     ymm2, [%2 + %5 + 64]
    vmovaps     ymm3, [%2 + %5 + 96]
    vsubps      ymm0, ymm0, [%3 + %5]
    vsubps      ymm1, ymm1, [%3 + %5 + 32]
    vsubps      ymm2, ymm2, [%3 + %5 + 64]
    vsubps      ymm3, ymm3, [%3 + %5 + 96]
    vaddps      ymm0, ymm0, [%1 + %5]
    vaddps      ymm1, ymm1, [%1 + %5 + 32]
    vaddps      ymm2, ymm2, [%1 + %5 + 64]
    vaddps      ymm3, ymm3, [%1 + %5 + 96]
    vmovaps     [%1 + %5], ymm0
    vmovaps     [%1 + %5 + 32], ymm1
    vmovaps     [%1 + %5 + 64], ymm2
    vmovaps     [%1 + %5 + 96], ymm3
    add         %5, 128
    cmp         %5, %6
    jb          %%block
%%tail:
    cmp         %5, %4
    jae         %%done
    vmovaps     ymm0, [%2 + %5]
    vsubps      ymm0, ymm0, [%3 + %5]
    vaddps      ymm0, ymm0, [%1 + %5]
    vmovaps     [%1 + %5], ymm0
    add         %5, 32
    jmp         %%tail
%%done:
%endmacro

; COPY_TEAM: best_team[0, k) = order[0, k).
; copies k int32_t values from order to best_team, using 8-wide AVX2 loads and stores.
%macro COPY_TEAM 0
    mov         rdx, [r15 + C_BEST]
    mov         r10d, [r15 + C_K]
    xor         eax, eax
%%full:
    cmp         r10d, 8
    jb          %%tail
    vmovdqu     ymm0, [r12 + rax*4]
    vmovdqu     [rdx + rax*4], ymm0
    add         eax, 8
    sub         r10d, 8
    jmp         %%full
%%tail:
    test        r10d, r10d
    jz          %%done
    neg         r10
    lea         r11, [TAIL_MASK + 32]
    vmovdqu     ymm1, [r11 + r10*4]         ; (k mod 8) leading all-ones lanes
    vpmaskmovd  ymm0, ymm1, [r12 + rax*4]
    vpmaskmovd  [rdx + rax*4], ymm1, ymm0
%%done:
%endmacro

section .text

global sa_avx2_cpu_supported
global sa_avx2_philox_x8
global sa_avx2_g_update
global sa_avx2_refresh
global sa_avx2_run_chain

; check for avx2 support, returns 1 in rax if supported, 0 otherwise
sa_avx2_cpu_supported:
    push        rbx
    xor         eax, eax
    cpuid
    cmp         eax, 7                      
    jb          .no
    mov         eax, 1
    xor         ecx, ecx
    cpuid
    and         ecx, (1 << 27) | (1 << 28)  
    cmp         ecx, (1 << 27) | (1 << 28)
    jne         .no
    xor         ecx, ecx
    xgetbv                                  
    and         eax, 0x6                    
    cmp         eax, 0x6
    jne         .no
    mov         eax, 7
    xor         ecx, ecx
    cpuid
    test        ebx, 1 << 5                 
    jz          .no
    mov         eax, 1
    pop         rbx
    ret
.no:
    xor         eax, eax
    pop         rbx
    ret

philox8_core:
    PHILOX_ROUND
%rep 9
    vpaddd      ymm4, ymm4, [PHILOX_W0]
    vpaddd      ymm5, ymm5, [PHILOX_W1]
    PHILOX_ROUND
%endrep
    ret

; generates a batch of 32 pseudo-random 32-bit integers, using the Philox4x32 algorithm, with 8-way parallelism.
sa_avx2_philox_x8:
    sub         rsp, 56                     ; 48 bytes for xmm6..8, 16-byte aligned
    vmovdqa     [rsp], xmm6
    vmovdqa     [rsp + 16], xmm7
    vmovdqa     [rsp + 32], xmm8

    BUILD_COUNTERS rcx, rax                 ; ymm0, ymm1 = c0, c1
    vmovq       xmm2, rdx
    vpbroadcastd ymm2, xmm2                 ; c2 = lo(chain_stream)
    mov         rax, rdx
    shr         rax, 32
    vmovq       xmm3, rax
    vpbroadcastd ymm3, xmm3                 ; c3 = hi(chain_stream)
    vmovq       xmm4, r8
    vpbroadcastd ymm4, xmm4                 ; k0 = lo(seed)
    mov         rax, r8
    shr         rax, 32
    vmovq       xmm5, rax
    vpbroadcastd ymm5, xmm5                 ; k1 = hi(seed)
    call        philox8_core
    vmovdqu     [r9], ymm0
    vmovdqu     [r9 + 32], ymm1
    vmovdqu     [r9 + 64], ymm2
    vmovdqu     [r9 + 96], ymm3

    vmovdqa     xmm6, [rsp]
    vmovdqa     xmm7, [rsp + 16]
    vmovdqa     xmm8, [rsp + 32]
    add         rsp, 56
    vzeroupper
    ret
; 


; exposes for A += B - C
sa_avx2_g_update:
    lea         r10, [r9*4]                 ; bytes
    G_UPDATE    rcx, rdx, r8, r10, r11, rax
    vzeroupper
    ret

; full state refesh for simulated annealing 
; recomputes the gradient vector across all candidates using AVX2
; recomputes the total scaler system energy using AVX2
refresh_core:
    push        rbx
    push        rsi
    push        rdi
    push        r12
    mov         r8,  [r15 + C_G]
    mov         r9,  [r15 + C_W]
    mov         r10, [r15 + C_LAMP]
    mov         r11, [r15 + C_ORDER]        ; the team is order[0, k)
    mov         rdx, [r15 + C_STRIDE]
    shl         rdx, 2                      ; bytes per row
    mov         edi, [r15 + C_K]
    vmovaps     ymm4, [SIGN_BIT]

    mov         r12, rdx
    and         r12, -128                   ; end of the 32-drug blocks
    xor         eax, eax                    ; byte offset of the current drug block
.block32:
    cmp         rax, r12
    jae         .block8
    vxorps      ymm0, ymm4, [r9 + rax]      ; -w, 32 drugs in 4 registers
    vxorps      ymm1, ymm4, [r9 + rax + 32]
    vxorps      ymm2, ymm4, [r9 + rax + 64]
    vxorps      ymm3, ymm4, [r9 + rax + 96]
    xor         esi, esi                    ; t
.block32_team:
    cmp         esi, edi
    jge         .block32_store
    movsxd      rcx, dword [r11 + rsi*4]    ; team[t]
    imul        rcx, rdx
    add         rcx, r10                    ; its row
    vaddps      ymm0, ymm0, [rcx + rax]
    vaddps      ymm1, ymm1, [rcx + rax + 32]
    vaddps      ymm2, ymm2, [rcx + rax + 64]
    vaddps      ymm3, ymm3, [rcx + rax + 96]
    inc         esi
    jmp         .block32_team
.block32_store:
    vmovaps     [r8 + rax], ymm0
    vmovaps     [r8 + rax + 32], ymm1
    vmovaps     [r8 + rax + 64], ymm2
    vmovaps     [r8 + rax + 96], ymm3
    add         rax, 128
    jmp         .block32

.block8:
    cmp         rax, rdx
    jae         .energy
    vxorps      ymm0, ymm4, [r9 + rax]
    xor         esi, esi
.block8_team:
    cmp         esi, edi
    jge         .block8_store
    movsxd      rcx, dword [r11 + rsi*4]
    imul        rcx, rdx
    add         rcx, r10
    vaddps      ymm0, ymm0, [rcx + rax]
    inc         esi
    jmp         .block8_team
.block8_store:
    vmovaps     [r8 + rax], ymm0
    add         rax, 32
    jmp         .block8

.energy:                                    ; O(k^2) scalar: same order as team_energy()
    vxorps      xmm0, xmm0, xmm0
    xor         esi, esi                    ; a
.energy_a:
    cmp         esi, edi
    jge         .energy_done
    movsxd      rcx, dword [r11 + rsi*4]    ; team[a]
    vsubss      xmm0, xmm0, [r9 + rcx*4]    ; energy -= w[team[a]]
    imul        rcx, rdx
    add         rcx, r10                    ; row team[a]
    lea         ebx, [rsi + 1]              ; b = a + 1
.energy_b:
    cmp         ebx, edi
    jge         .energy_next
    movsxd      rax, dword [r11 + rbx*4]    ; team[b]
    vaddss      xmm0, xmm0, [rcx + rax*4]   ; energy += lam_p[team[a]][team[b]]
    inc         ebx
    jmp         .energy_b
.energy_next:
    inc         esi
    jmp         .energy_a
.energy_done:
    vmovss      [r15 + C_ENERGY], xmm0
    pop         r12
    pop         rdi
    pop         rsi
    pop         rbx
    ret

wrapper for refresh core
sa_avx2_refresh:
    push        r15
    mov         r15, rcx                    ; the chain
    call        refresh_core
    pop         r15
    vzeroupper
    ret


; stack frame memory layout of the SA Kernel
SPECULATE_MAX_LAST_ACCEPTS equ 1

; Stack frame, offsets from the 32-byte aligned rsp
%define F_A         0       ; 8 x int32   team slot a of each lane
%define F_B         32      ; 8 x int32   outsider slot b
%define F_I         64      ; 8 x int32   i = order[a], the drug that would leave
%define F_J         96      ; 8 x int32   j = order[b], the drug that would join
%define F_D         128     ; 8 x float   delta E
%define F_T         160     ; 8 x float   thr = T * L[r2 >> 20]
%define F_NK        192     ; 8 x int32   n - k
%define F_K0        224     ; 8 x int32   lo(seed)
%define F_K1        256     ; 8 x int32   hi(seed)
%define F_XMM       288     ; 160 bytes   xmm6..xmm15 (Win64)
%define F_COUNT     448     ; int32       proposals in this batch (1..8)
%define F_P         452     ; int32       first lane not yet decided
%define F_SINCE     456     ; int32       accepts since the last refresh
%define F_TEMP      460     ; float       temperature of this sweep
%define F_BATCH_ACC 464     ; int32       accepts in this batch
%define F_LAST_ACC  468     ; int32       accepts in the previous batch
%define F_SIZE      480

sa_avx2_run_chain:
    push        rbp
    mov         rbp, rsp
    push        rbx
    push        rsi
    push        rdi
    push        r12
    push        r13
    push        r14
    push        r15
    sub         rsp, F_SIZE + 32
    and         rsp, -32
    SAVE_XMM6_15 rsp + F_XMM

    mov         r15, rcx                    ; the chain
    mov         r14, [r15 + C_LAMP]
    mov         r13, [r15 + C_G]
    mov         r12, [r15 + C_ORDER]

    vpbroadcastd ymm9,  [r15 + C_K]
    vpbroadcastd ymm10, [r15 + C_STRIDE]    ; low dword; stride * n < 2^31 (sa_avx2.c)
    vpbroadcastd ymm14, [r15 + C_CHAIN]
    mov         eax, [r15 + C_N]
    sub         eax, [r15 + C_K]
    vmovd       xmm0, eax
    vpbroadcastd ymm0, xmm0
    vmovdqa     [rsp + F_NK], ymm0
    vpbroadcastd ymm0, [r15 + C_SEED]
    vmovdqa     [rsp + F_K0], ymm0
    vpbroadcastd ymm0, [r15 + C_SEED + 4]
    vmovdqa     [rsp + F_K1], ymm0
    mov         qword [r15 + C_ACCEPTS], 0
    mov         dword [rsp + F_SINCE], 0
    mov         dword [rsp + F_BATCH_ACC], 0
    xor         ebx, ebx                    ; step = 0
    xor         edi, edi                    ; sweep = 0

.sweep:
    cmp         edi, [r15 + C_SWEEPS]
    jge         .finish
    mov         rax, [r15 + C_TEMPS]
    mov         eax, [rax + rdi*4]
    mov         [rsp + F_TEMP], eax         ; T for this sweep
    xor         esi, esi                    ; q = 0

.batch:
    mov         eax, [r15 + C_N]
    sub         eax, esi                    ; proposals left in this sweep
    jle         .sweep_done
    mov         ecx, 8
    cmp         eax, ecx
    cmova       eax, ecx                    ; count = min(8, left)
    mov         [rsp + F_COUNT], eax
    mov         eax, [rsp + F_BATCH_ACC]
    mov         [rsp + F_LAST_ACC], eax
    mov         dword [rsp + F_BATCH_ACC], 0
    mov         dword [rsp + F_P], 0

    ; -Philox
    BUILD_COUNTERS rbx, rax                 ; ymm0, ymm1 = lo, hi of step + lane
    vmovdqa     ymm2, ymm14                 ; c2 = chain
    vpxor       ymm3, ymm3, ymm3            ; c3 = STREAM_PROPOSAL = 0
    vmovdqa     ymm4, [rsp + F_K0]
    vmovdqa     ymm5, [rsp + F_K1]
    call        philox8_core                ; ymm0, ymm1, ymm2 = r0, r1, r2

    ; define slots and acceptance thresholds on 8 lanes
    MULHI_U32   ymm13, ymm0, ymm9, ymm6             ; a = (r0 * k) >> 32
    MULHI_U32   ymm12, ymm1, [rsp + F_NK], ymm6     ; (r1 * (n - k)) >> 32
    vpaddd      ymm12, ymm12, ymm9                  ; b = k + that
    vpsrld      ymm2, ymm2, ACCEPT_TABLE_SHIFT      ; r2 >> 20
    vpcmpeqd    ymm7, ymm7, ymm7
    mov         rax, [r15 + C_LNU]
    vxorps      ymm11, ymm11, ymm11
    vgatherdps  ymm11, [rax + ymm2*4], ymm7         ; L[r2 >> 20]
    vbroadcastss ymm6, [rsp + F_TEMP]
    vmulps      ymm11, ymm6, ymm11                  ; thr = T * L[r2 >> 20]
    vmovdqa     [rsp + F_A], ymm13
    vmovdqa     [rsp + F_B], ymm12
    vmovaps     [rsp + F_T], ymm11

    cmp         dword [rsp + F_LAST_ACC], SPECULATE_MAX_LAST_ACCEPTS
    ja          .walk

    ; speculative pass
    vpcmpeqd    ymm7, ymm7, ymm7
    vpxor       ymm0, ymm0, ymm0
    vpgatherdd  ymm0, [r12 + ymm13*4], ymm7         ; i = order[a]
    vpcmpeqd    ymm7, ymm7, ymm7
    vpxor       ymm1, ymm1, ymm1
    vpgatherdd  ymm1, [r12 + ymm12*4], ymm7         ; j = order[b]
    vpcmpeqd    ymm7, ymm7, ymm7
    vxorps      ymm2, ymm2, ymm2
    vgatherdps  ymm2, [r13 + ymm0*4], ymm7          ; g[i]
    vpcmpeqd    ymm7, ymm7, ymm7
    vxorps      ymm3, ymm3, ymm3
    vgatherdps  ymm3, [r13 + ymm1*4], ymm7          ; g[j]
    vpmulld     ymm4, ymm0, ymm10
    vpaddd      ymm4, ymm4, ymm1                    ; i * stride + j
    vpcmpeqd    ymm7, ymm7, ymm7
    vxorps      ymm5, ymm5, ymm5
    vgatherdps  ymm5, [r14 + ymm4*4], ymm7          ; lambda * P[i][j]
    vsubps      ymm6, ymm3, ymm2                    ; g[j] - g[i]
    vsubps      ymm6, ymm6, ymm5                    ;   - lambda*P[i][j] = delta E
    vxorps      ymm7, ymm7, ymm7
    vcmpps      ymm7, ymm6, ymm7, CMP_LE_OQ         ; delta <= 0
    vcmpps      ymm8, ymm6, ymm11, CMP_LT_OQ        ; delta < T * L
    vorps       ymm7, ymm7, ymm8                    ; accept
    vmovmskps   eax, ymm7                           ; one bit per lane
    mov         ecx, [rsp + F_COUNT]
    mov         edx, 1
    shl         edx, cl
    dec         edx
    and         eax, edx                    ; only lanes < count are proposals
    jz          .batch_done                 ; all rejected: the whole batch is decided
    tzcnt       ecx, eax                    ; m = first accepted lane; 0..m-1 rejected
    vmovdqa     [rsp + F_I], ymm0
    vmovdqa     [rsp + F_J], ymm1
    vmovaps     [rsp + F_D], ymm6
    mov         r8d, [rsp + F_I + rcx*4]
    mov         r9d, [rsp + F_J + rcx*4]
    vmovss      xmm3, [rsp + F_D + rcx*4]
    jmp         .commit

    
.walk:
    mov         ecx, [rsp + F_P]
.walk_lane:
    cmp         ecx, [rsp + F_COUNT]
    jae         .batch_done
    mov         eax, [rsp + F_A + rcx*4]
    mov         r8d, [r12 + rax*4]          ; i = order[a]
    mov         eax, [rsp + F_B + rcx*4]
    mov         r9d, [r12 + rax*4]          ; j = order[b]
    vmovss      xmm3, [r13 + r9*4]
    vsubss      xmm3, xmm3, [r13 + r8*4]    ; g[j] - g[i]
    mov         rax, r8
    imul        rax, [r15 + C_STRIDE]
    add         rax, r9
    vsubss      xmm3, xmm3, [r14 + rax*4]   ;   - lambda*P[i][j] = delta E
    vxorps      xmm1, xmm1, xmm1
    vcomiss     xmm1, xmm3
    jae         .commit                     ; 0 >= delta (false for NaN, as in C)
    vmovss      xmm1, [rsp + F_T + rcx*4]
    vcomiss     xmm1, xmm3
    ja          .commit                     ; T * L > delta
    inc         ecx
    jmp         .walk_lane

    ; ---- (4) commit lane m (commit_swap in sa_scalar.c) ----
    ; rcx = m, r8 = i (leaves the team), r9 = j (joins it), xmm3 = delta E
.commit:
    mov         eax, [rsp + F_A + rcx*4]
    mov         [r12 + rax*4], r9d          ; order[a] = j
    mov         eax, [rsp + F_B + rcx*4]
    mov         [r12 + rax*4], r8d          ; order[b] = i

    vmovss      xmm0, [r15 + C_ENERGY]
    vaddss      xmm0, xmm0, xmm3
    vmovss      [r15 + C_ENERGY], xmm0      ; energy += delta
    vmovss      xmm1, [r15 + C_BESTE]
    vcomiss     xmm1, xmm0
    jbe         .not_best                   ; unless best_energy > energy
    vmovss      [r15 + C_BESTE], xmm0
    COPY_TEAM
.not_best:

    lea         rax, [rbx + rcx]            ; this proposal's step number
    cmp         rax, [r15 + C_TSTEPS]
    jae         .not_traced
    mov         rdx, [r15 + C_TRACE]
    mov         byte [rdx + rax], 1
.not_traced:
    inc         qword [r15 + C_ACCEPTS]
    inc         dword [rsp + F_BATCH_ACC]
    inc         ecx
    mov         [rsp + F_P], ecx            ; the lanes after m are next

    mov         r10, [r15 + C_STRIDE]
    shl         r10, 2                      ; bytes per row
    mov         rax, r9
    imul        rax, r10
    add         rax, r14                    ; row j of lambda*P
    mov         rdx, r8
    imul        rdx, r10
    add         rdx, r14                    ; row i
    G_UPDATE    r13, rax, rdx, r10, r11, rcx    ; g += lambda*P[j,:] - lambda*P[i,:]

    inc         dword [rsp + F_SINCE]
    cmp         dword [rsp + F_SINCE], REFRESH_EVERY_ACCEPTS
    jne         .walk
    mov         dword [rsp + F_SINCE], 0
    call        refresh_core                ; rebuild g and E from scratch (float drift)
    jmp         .walk

.batch_done:
    mov         eax, [rsp + F_COUNT]
    add         rbx, rax
    add         esi, eax
    jmp         .batch

.sweep_done:
    inc         edi
    jmp         .sweep

.finish:
    RESTORE_XMM6_15 rsp + F_XMM
    lea         rsp, [rbp - 56]
    pop         r15
    pop         r14
    pop         r13
    pop         r12
    pop         rdi
    pop         rsi
    pop         rbx
    pop         rbp
    vzeroupper
    ret
