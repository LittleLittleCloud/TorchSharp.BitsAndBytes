@echo off
setlocal enabledelayedexpansion

REM Get configuration arugment from build.proj
set "configuration=%1"

REM Use the configuration argument to determine the build OS and architecture
echo Building for configuration %configuration%
echo Building for cuda_version %cuda_version%
echo Building for build_os %build_os%

set build_capability=50;52;60;61;70;75;80;86;89;90;100;120
set remove_for_11_7=;89;90;100;120
set remove_for_11_8=;100;120
set remove_for_lt_12_7=;100;120

if "%cuda_version%" == "11.7.*" (
    set build_capability=!build_capability:%remove_for_11_7%=!
)
if "%cuda_version%" == "11.8.*" (
    set build_capability=!build_capability:%remove_for_11_8%=!
)
if "%cuda_version%" lss "12.7" (
    set build_capability=!build_capability:%remove_for_lt_12_7%=!
    set build_capability=!build_capability:%remove_for_lt_12_7:~1%=!
)
if "%build_os:~0,7%" == "windows" (
    python -m pip install ninja
)

if "%build_os:~0,6%" == "ubuntu" (
    set image=nvidia/cuda:%cuda_version%-devel-ubuntu22.04
    echo Using image %image%
    docker run --platform "linux/%build_arch%" -i -w /src -v "%cd%:/src" "%image%" sh -c ^
        "apt-get update && ^
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends cmake && ^
        cmake -DPTXAS_VERBOSE=1 -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY=\"%build_capability%\" . && ^
        cmake --build ."
) else (
    pip install cmake==3.28.3
    cmake -G Ninja -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY="%build_capability%" -DCMAKE_BUILD_TYPE=%Configuration% -S ..\bitsandbytes
    cmake --build . --config %Configuration%
)