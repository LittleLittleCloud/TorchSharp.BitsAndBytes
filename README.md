# TorchSharp.BitsAndBytes
The `TorchSharp.BitsAndBytes` is a C# binding library for [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) library from Huggingface. It provides 4Bit and 8Bit quantization for TorchSharp models.

## Usage
### 4Bit Quantization && Dequantization
>[!NOTE]
> 4Bit quantization is only available for CUDA devices.
```csharp
var input = torch.rand([dim * 4, dim], dtype: ScalarType.Float32).cuda(); // FP32 tensor, must be on cuda device
string quantizedDType = "fp4"; // Available options: "fp4", "nf4"
int blockSize = 64; // can be [64, 128, 256, 512, 1024]

// Quantize to 4Bit
(var quantizedTensor, var absMax, blockSize, var n) = BitsAndByteUtils.Quantize4Bit(input, quantizedDType, blockSize);

// Dequantize to FP32
var dequantizedTensor = BitsAndByteUtils.Dequantize4Bit(quantiedTensor, absMax, input.dtype, quantizedDType, n, input.shape, blockSize);
```

For more examples, please refer to the [Benchmark](#Benchmark) section.

## Benchmark
```

BenchmarkDotNet v0.14.0, Windows 11 (10.0.26100.3037)
Intel Core i9-14900K, 1 CPU, 32 logical and 24 physical cores
.NET SDK 9.0.102
  [Host]     : .NET 8.0.12 (8.0.1224.60305), X64 RyuJIT AVX2
  DefaultJob : .NET 8.0.12 (8.0.1224.60305), X64 RyuJIT AVX2


```
| Method         | Mean        | Error     | StdDev    |
|--------------- |------------:|----------:|----------:|
| Quantize4Bit   |   536.35 μs | 12.164 μs | 35.290 μs |
| Dequantize4Bit | 2,257.89 μs | 44.542 μs | 51.294 μs |
| GEMV_4Bit_FP4  |    84.16 μs |  1.673 μs |  3.223 μs |
| GEMV_4Bit_NF4  |    82.69 μs |  4.329 μs | 12.629 μs |
| GEMV_FP32      |    49.59 μs |  0.975 μs |  2.035 μs |
| GEMM_INT8      | 2,994.86 μs | 12.144 μs | 11.360 μs |
| GEMM_FP32      | 4,495.49 μs | 35.264 μs | 32.986 μs |
