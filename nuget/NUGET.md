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