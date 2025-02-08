setlocal enabledelayedexpansion

set "configuration=%1"
set "build_arch=%2"
set "build_os=%3"

REM Use the configuration argument to determine the build OS and architecture
echo Building for configuration %configuration%
echo Building for build_arch %build_arch%
echo Building for build_os %build_os%

pip install cmake==3.28.3

if "%build_os:~0,6%" == "ubuntu" (
    if "%build_arch%" == "aarch64" (
        :: Allow cross-compile on aarch64
        sudo apt-get update
        sudo apt-get install -y gcc-aarch64-linux-gnu binutils-aarch64-linux-gnu g++-aarch64-linux-gnu
        cmake -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ -DCOMPUTE_BACKEND=cpu .
    )
) else if "%build_os:~0,5%" == "macos" (
    if "%build_arch%" == "aarch64" (
        cmake -DCMAKE_OSX_ARCHITECTURES=arm64 -DCOMPUTE_BACKEND=cpu .
    )
) else (
    cmake -DCOMPUTE_BACKEND=cpu ..\bitsandbytes
)
cmake --build . --config %configuration%