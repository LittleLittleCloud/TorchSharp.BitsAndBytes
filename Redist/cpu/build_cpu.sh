#!/bin/bash
set -xeuo pipefail

build_config=${1:-Release}
build_arch=${2}
build_os=${3}

# Check if cmake is installed and its version is >= 3.28.3
if ! command -v cmake &> /dev/null || [[ "$(cmake --version | head -n1 | awk '{print $3}')" < "3.28.3" ]]; then
    pip install cmake==3.28.3
fi

if [ "${build_os:0:6}" == ubuntu ] && [ "${build_arch}" == aarch64 ]; then
	# Allow cross-compile on aarch64
	sudo apt-get update
	sudo apt-get install -y gcc-aarch64-linux-gnu binutils-aarch64-linux-gnu g++-aarch64-linux-gnu
	cmake -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ -DCOMPUTE_BACKEND=cpu ../bitsandbytes
elif [ "${build_os:0:3}" == mac ]; then
	cmake -DCMAKE_OSX_ARCHITECTURES=arm64 -DCOMPUTE_BACKEND=cpu ../bitsandbytes
else
	cmake -DCOMPUTE_BACKEND=cpu ../bitsandbytes
fi
cmake --build . --config "${build_config}"
