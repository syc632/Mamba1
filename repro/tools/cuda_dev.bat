@echo off
rem ---------------------------------------------------------------------------
rem Run a command inside the MSVC x64 dev environment + extracted CUDA toolkit.
rem   usage: cuda_dev.bat <command> [args...]
rem ---------------------------------------------------------------------------
set "CUDA_HOME=D:/cuda/v13.2"
set "CUDA_PATH=D:/cuda/v13.2"
set "CUDA_PATH_V13_2=D:/cuda/v13.2"
set "PATH=D:/cuda/v13.2/bin;%PATH%"
set "DISTUTILS_USE_SDK=1"
set "MSSdk=1"
rem CCCL (cub/thrust) requires the conforming preprocessor; nvcc picks these up
rem automatically for any invocation, including from torch cpp_extension.
set "NVCC_APPEND_FLAGS=-std=c++17 -Xcompiler /Zc:preprocessor -Xcompiler /wd4819"
set "NVCC_PREPEND_FLAGS=-std=c++17 -Xcompiler /Zc:preprocessor -Xcompiler /wd4819"

call "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat" >nul 2>&1

rem vcvars blanks these when its registry lookup fails, so set them AFTER.
set "WindowsSdkDir=C:/Program Files (x86)\Windows Kits\10"
set "WindowsSDKVersion=10.0.26100.0"
set "UCRTVersion=10.0.26100.0"
set "UniversalCRTSdkDir=C:/Program Files (x86)\Windows Kits\10"
set "SDK_INC=%WindowsSdkDir%\Include\%WindowsSDKVersion%"
set "SDK_LIB=%WindowsSdkDir%\Lib\%WindowsSDKVersion%"
set "INCLUDE=%SDK_INC%\ucrt;%SDK_INC%\um;%SDK_INC%\shared;%SDK_INC%\winrt;%SDK_INC%\cppwinrt;%INCLUDE%"
set "LIB=%SDK_LIB%\ucrt\x64;%SDK_LIB%\um\x64;%LIB%"
set "PATH=%WindowsSdkDir%\bin\%WindowsSDKVersion%\x64;%PATH%"

echo [cuda_dev] CUDA_HOME=%CUDA_HOME%
%*
exit /b %ERRORLEVEL%
