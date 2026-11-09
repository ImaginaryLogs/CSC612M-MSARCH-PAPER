@echo off
rem run.bat -- double-click to run the AD drug-combination solvers on Windows.
rem
rem   run.bat                   a menu: compare scalar vs AVX2, real data, benchmark,
rem                             tests, rebuild. Offers the first-time setup if needed.
rem   run.bat <main.py options> passes the options straight to python\main.py, e.g.
rem       run.bat --synthetic 1000 --platforms scalar avx2 --k 5 --t0 2.0 --t1 0.02
rem
rem First-time setup (done for you when missing) needs Python 3.11+ with the "py"
rem launcher, Visual Studio with C++ and CMake, and NASM for the AVX2 module.
setlocal
cd /d "%~dp0"
set "PY=%CD%\.venv\Scripts\python.exe"

if not exist "%PY%" goto :offer_setup
if not exist "python\module_2_avx*.pyd" goto :offer_build
if not "%~1"=="" goto :pass_through

:menu
cls
echo AD drug-combination solver
echo.
echo   1  Compare scalar vs AVX2 on a random problem
echo   2  Run on the real Alzheimer's data (downloads it the first time)
echo   3  Speed benchmark tables
echo   4  Run all tests
echo   5  Rebuild (after changing C or assembly code)
echo   Q  Quit
echo.
set "CHOICE="
set /p "CHOICE=Choose 1-5 or Q: "
if /i "%CHOICE%"=="1" goto :synthetic
if /i "%CHOICE%"=="2" goto :real_data
if /i "%CHOICE%"=="3" goto :benchmark
if /i "%CHOICE%"=="4" goto :tests
if /i "%CHOICE%"=="5" goto :rebuild
if /i "%CHOICE%"=="Q" exit /b 0
goto :menu

:synthetic
set "N=512"
set "K=5"
echo.
set /p "N=Number of drugs [512]: "
set /p "K=Drugs to pick [5]: "
echo.
"%PY%" python\main.py --synthetic %N% --platforms scalar avx2 --k %K% --t0 2.0 --t1 0.02
goto :done

:real_data
echo.
"%PY%" python\main.py --platforms exact_cpu scalar avx2 --k 2 --t0 2.0 --t1 0.02
goto :done

:benchmark
if not exist "build\Release\sa_bench.exe" goto :offer_build
echo.
build\Release\sa_bench.exe --quick
goto :done

:tests
echo.
call :find_cmake
if not defined CMAKE_BIN echo CMake not found: skipping the C tests.
if defined CMAKE_BIN "%CMAKE_BIN%ctest.exe" --test-dir build -C Release
"%PY%" -m pytest tests -q -p no:cacheprovider
goto :done

:rebuild
echo.
call :find_cmake
if not defined CMAKE_BIN goto :no_cmake
"%CMAKE_BIN%cmake.exe" -S . -B build -DAD_BUILD_PYTHON=ON -DPython_EXECUTABLE="%PY%" || goto :failed
"%CMAKE_BIN%cmake.exe" --build build --config Release || goto :failed
copy /y build\Release\*.pyd python\ >nul || goto :failed
echo.
echo Built, and the modules are copied into python\.
goto :done

:pass_through
"%PY%" python\main.py %*
exit /b %ERRORLEVEL%

:offer_setup
echo The Python environment (.venv) is not set up yet.
set "ANSWER=Y"
set /p "ANSWER=Set it up now? This downloads Python packages [Y/n]: "
if /i not "%ANSWER%"=="Y" exit /b 1
where py >nul 2>nul || goto :no_python
echo.
py -3 -m venv .venv || goto :failed
"%PY%" -m pip install numpy pandas requests networkx pytest pybind11 || goto :failed
goto :rebuild

:offer_build
echo The solvers are not built yet.
set "ANSWER=Y"
set /p "ANSWER=Build them now? [Y/n]: "
if /i not "%ANSWER%"=="Y" exit /b 1
goto :rebuild

:find_cmake
rem Sets CMAKE_BIN to the folder holding cmake.exe and ctest.exe (with a trailing \):
rem CMake on PATH first, else the copy that ships with Visual Studio.
set "CMAKE_BIN="
for /f "delims=" %%i in ('where cmake 2^>nul') do if not defined CMAKE_BIN set "CMAKE_BIN=%%~dpi"
if defined CMAKE_BIN exit /b 0
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" exit /b 0
set "VSDIR="
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -property installationPath`) do set "VSDIR=%%i"
set "VSCMAKE=%VSDIR%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\"
if exist "%VSCMAKE%cmake.exe" set "CMAKE_BIN=%VSCMAKE%"
exit /b 0

:no_python
echo The Python launcher "py" was not found. Install Python 3.11 or newer from
echo https://www.python.org/downloads/ and run this again.
goto :done_then_exit

:no_cmake
echo CMake was not found. Install Visual Studio with "Desktop development with C++"
echo (it includes CMake), and NASM from https://www.nasm.us for the AVX2 module.
goto :done

:failed
echo.
echo Something above failed.
goto :done

:done
echo.
pause
if not exist "%PY%" exit /b 1
if not exist "python\module_2_avx*.pyd" exit /b 1
goto :menu

:done_then_exit
echo.
pause
exit /b 1
