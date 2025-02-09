using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;

namespace TorchSharp.BitsAndBytes;

public class BitsAndByteUtils
{
    private static readonly Lazy<Dictionary<(string, string, int), Tensor>> _4bitTypeCache = new Lazy<Dictionary<(string, string, int), Tensor>>(); 
    public static (
            Tensor quantizedTensor,
            Tensor absMax,
            int blockSize,
            int n
            )
            Quantize4Bit(
            Tensor tensor, // input tensor
            string quantizedDType = "fp4", // quantized data type, must be one of "fp4", "nf4"
            int blockSize = 64 // block size
            )
    {
        var n = (int)torch.numel(tensor);
        var blocks = (int)Math.Ceiling((double)n / blockSize);
        var absMax = torch.zeros([blocks], dtype: torch.float32).cuda();
        var mod = 2;
        var quantizedTensor = torch.zeros([(n + 1) / mod, 1], dtype: ScalarType.Byte).cuda();
        if (tensor.dtype == ScalarType.Float32)
        {
            if (quantizedDType == "fp4")
            {
                BitsAndBytes.BitsAndBytesCudaNative.cquantize_blockwise_fp32_fp4(
                IntPtr.Zero,
                LibTorchNativeMethod.THSStorage_data_ptr(tensor.Handle),
                LibTorchNativeMethod.THSStorage_data_ptr(absMax.Handle),
                LibTorchNativeMethod.THSStorage_data_ptr(quantizedTensor.Handle),
                blockSize,
                n);
            }
            else
            {
                BitsAndBytes.BitsAndBytesCudaNative.cquantize_blockwise_fp32_nf4(
                IntPtr.Zero,
                LibTorchNativeMethod.THSStorage_data_ptr(tensor.Handle),
                LibTorchNativeMethod.THSStorage_data_ptr(absMax.Handle),
                LibTorchNativeMethod.THSStorage_data_ptr(quantizedTensor.Handle),
                blockSize,
                n);
            }
        }
        else if (tensor.dtype == ScalarType.BFloat16)
        {
            if (quantizedDType == "fp4")
            {
                BitsAndBytes.BitsAndBytesCudaNative.cquantize_blockwise_bf16_fp4(
                IntPtr.Zero,
                LibTorchNativeMethod.THSStorage_data_ptr(tensor.Handle),
                LibTorchNativeMethod.THSStorage_data_ptr(absMax.Handle),
                LibTorchNativeMethod.THSStorage_data_ptr(quantizedTensor.Handle),
                blockSize,
                n);
            }
            else
            {
                BitsAndBytes.BitsAndBytesCudaNative.cquantize_blockwise_bf16_nf4(
                IntPtr.Zero,
                LibTorchNativeMethod.THSStorage_data_ptr(tensor.Handle),
                LibTorchNativeMethod.THSStorage_data_ptr(absMax.Handle),
                LibTorchNativeMethod.THSStorage_data_ptr(quantizedTensor.Handle),
                blockSize,
                n);
            }

        }
        else if (tensor.dtype == ScalarType.Float16)
        {
            if (quantizedDType == "fp4")
            {
                BitsAndBytes.BitsAndBytesCudaNative.cquantize_blockwise_fp16_fp4(
                IntPtr.Zero,
                LibTorchNativeMethod.THSStorage_data_ptr(tensor.Handle),
                LibTorchNativeMethod.THSStorage_data_ptr(absMax.Handle),
                LibTorchNativeMethod.THSStorage_data_ptr(quantizedTensor.Handle),
                blockSize,
                n);
            }
            else
            {
                BitsAndBytes.BitsAndBytesCudaNative.cquantize_blockwise_fp16_nf4(
                IntPtr.Zero,
                LibTorchNativeMethod.THSStorage_data_ptr(tensor.Handle),
                LibTorchNativeMethod.THSStorage_data_ptr(absMax.Handle),
                LibTorchNativeMethod.THSStorage_data_ptr(quantizedTensor.Handle),
                blockSize,
                n);
            }
        }
        else
        {
            throw new NotImplementedException();
        }
        return (quantizedTensor, absMax, blockSize, n);
    }

    public static Tensor Dequantize4Bit(
        Tensor tensor, // quantized tensor
        Tensor absMax, // absMax tensor
        ScalarType originalDType, // original data type
        string quantizedDType, // quantized data type, must be one of "fp4", "nf4"
        int n,
        long[] originalShape,
        int blockSize = 64, // block size
        ScalarType quantStorageDType = ScalarType.Byte // quantized storage data type
        )
    {
        var dequantizedTensor = torch.zeros(originalShape, dtype: originalDType).cuda();
        if (originalDType == ScalarType.Float32)
        {
            if (quantizedDType == "fp4")
            {
                BitsAndBytes.BitsAndBytesCudaNative.cdequantize_blockwise_fp32_fp4(
                    IntPtr.Zero,
                    LibTorchNativeMethod.THSStorage_data_ptr(tensor.Handle),
                    LibTorchNativeMethod.THSStorage_data_ptr(absMax.Handle),
                    LibTorchNativeMethod.THSStorage_data_ptr(dequantizedTensor.Handle),
                    blockSize,
                    n,
                    IntPtr.Zero);
            }
            else
            {
                BitsAndBytes.BitsAndBytesCudaNative.cdequantize_blockwise_fp32_nf4(
                    IntPtr.Zero,
                    LibTorchNativeMethod.THSStorage_data_ptr(tensor.Handle),
                    LibTorchNativeMethod.THSStorage_data_ptr(absMax.Handle),
                    LibTorchNativeMethod.THSStorage_data_ptr(dequantizedTensor.Handle),
                    blockSize,
                    n,
                    IntPtr.Zero);
            }
        }
        else if (originalDType == ScalarType.BFloat16)
        {
            if (quantizedDType == "fp4")
            {
                BitsAndBytes.BitsAndBytesCudaNative.cdequantize_blockwise_bf16_fp4(
                    IntPtr.Zero,
                    LibTorchNativeMethod.THSStorage_data_ptr(tensor.Handle),
                    LibTorchNativeMethod.THSStorage_data_ptr(absMax.Handle),
                    LibTorchNativeMethod.THSStorage_data_ptr(dequantizedTensor.Handle),
                    blockSize,
                    n,
                    IntPtr.Zero);
            }
            else
            {
                BitsAndBytes.BitsAndBytesCudaNative.cdequantize_blockwise_bf16_nf4(
                    IntPtr.Zero,
                    LibTorchNativeMethod.THSStorage_data_ptr(tensor.Handle),
                    LibTorchNativeMethod.THSStorage_data_ptr(absMax.Handle),
                    LibTorchNativeMethod.THSStorage_data_ptr(dequantizedTensor.Handle),
                    blockSize,
                    n,
                    IntPtr.Zero);
            }
        }
        else if (originalDType == ScalarType.Float16)
        {
            if (quantizedDType == "fp4")
            {
                BitsAndBytes.BitsAndBytesCudaNative.cdequantize_blockwise_fp16_fp4(
                    IntPtr.Zero,
                    LibTorchNativeMethod.THSStorage_data_ptr(tensor.Handle),
                    LibTorchNativeMethod.THSStorage_data_ptr(absMax.Handle),
                    LibTorchNativeMethod.THSStorage_data_ptr(dequantizedTensor.Handle),
                    blockSize,
                    n,
                    IntPtr.Zero);
            }
            else
            {
                BitsAndBytes.BitsAndBytesCudaNative.cdequantize_blockwise_fp16_nf4(
                    IntPtr.Zero,
                    LibTorchNativeMethod.THSStorage_data_ptr(tensor.Handle),
                    LibTorchNativeMethod.THSStorage_data_ptr(absMax.Handle),
                    LibTorchNativeMethod.THSStorage_data_ptr(dequantizedTensor.Handle),
                    blockSize,
                    n,
                    IntPtr.Zero);
            }
        }

        return dequantizedTensor;
    }



    public static Tensor Get4BitType(string typename, string device = "cuda", int blocksize = 64)
    {
        if (_4bitTypeCache.Value.TryGetValue((typename, device, blocksize), out var cachedTensor))
        {
            return cachedTensor;
        }
        float[] data = [];

        if (typename == "nf4")
        {
            // Implements the NF4 data type.
            // Constructs a quantization data type where each bin has equal area under a standard normal distribution N(0, 1) that
            // is normalized into the range [-1, 1].
            data = new float[]
            {
                -1.0f,
                -0.6961928f,
                -0.5250731f,
                -0.3949175f,
                -0.2844414f,
                -0.1847734f,
                -0.09105004f,
                0.0f,
                0.0795803f,
                0.1609302f,
                0.2461123f,
                0.3379152f,
                0.4407098f,
                0.562617f,
                0.7229568f,
                1.0f
            };
        }
        else if (typename == "fp4")
        {
            data = new float[]
            {
                0.0f, 0.0625f, 8.0f, 12.0f, 4.0f, 6.0f, 2.0f, 3.0f,
                -0.0f, -0.0625f, -8.0f, -12.0f, -4.0f, -6.0f, -2.0f, -3.0f
            };
        }
        else if (typename == "int4")
        {
            data = new float[]
            {
                7, 6, 5, 4, 3, 2, 1, 0, -0, -1, -2, -3, -4, -5, -6, -7
            };
        }
        else if (typename == "af4")
        {
            if (blocksize == 64)
            {
                data = new float[]
                {
                    -1.0f, -0.69441008f, -0.51243739f, -0.3736951f, -0.25607552f, -0.14982478f, -0.04934812f, 0.0f,
                    0.04273164f, 0.12934483f, 0.21961274f, 0.31675666f, 0.42563882f, 0.55496234f, 0.72424863f, 1.0f
                };
                Array.Reverse(data);
            }
            else
            {
                throw new NotImplementedException("4-bit AbnormalFloats currently only support blocksize 64.");
            }
        }

        if (data == null)
        {
            throw new NotImplementedException($"Typename {typename} not supported");
        }

        var tensor = torch.tensor(data, device: device);
        tensor.div_(tensor.abs().max());

        if (tensor.numel() != 16)
        {
            throw new Exception("Tensor does not have 16 elements.");
        }

        _4bitTypeCache.Value[(typename, device, blocksize)] = tensor;
        return tensor;
    }

    public static Tensor Gemv4Bit(
        Tensor input,
        Tensor quantizedWeight,
        long[] originalWeightShape,
        Tensor absMax,
        int blockSize,
        string quantizedDType) // quantized data type, must be one of "fp4", "nf4"
    {
        var inputShape = input.IntShape();
        if (input.numel() != inputShape[^1])
        {
            throw new ArgumentException("'Dimensions of A are invalid. Must be a vector with the leading dimensions of \"1\", e.g. [1, 1, 2048]'");
        }
        var batch = inputShape[0];
        var m = (int)originalWeightShape[0];
        var k = (int)originalWeightShape[1];
        var lda = (int)originalWeightShape[0];
        var ldc = (int)originalWeightShape[0];
        var ldb = (inputShape[^1] + 1) / 2;
        Tensor output;
        if (input.shape.Length == 3)
        {
            output = torch.zeros([batch, inputShape[1], originalWeightShape[0]], dtype: input.dtype).cuda();
        }
        else
        {
            output = torch.zeros([batch, originalWeightShape[0]], dtype: input.dtype).cuda();
        }

        // quantize weight
        var code = Get4BitType(quantizedDType, "cuda", blockSize);

        if (input.dtype == ScalarType.Float32)
        { 
            BitsAndBytesCudaNative.cgemm_4bit_inference_naive_fp32(
                m: m,
                n: batch,
                k: k,
                A: LibTorchNativeMethod.THSStorage_data_ptr(input.Handle),
                B: LibTorchNativeMethod.THSStorage_data_ptr(quantizedWeight.T.Handle),
                absmax: LibTorchNativeMethod.THSStorage_data_ptr(absMax.Handle),
                datatype: LibTorchNativeMethod.THSStorage_data_ptr(code.Handle),
                output: LibTorchNativeMethod.THSStorage_data_ptr(output.Handle),
                lda: lda,
                ldb: ldb,
                ldc: ldc,
                blocksize: blockSize,
                stream: IntPtr.Zero);
        }
        else if (input.dtype == ScalarType.Float16)
        {
            BitsAndBytesCudaNative.cgemm_4bit_inference_naive_fp16(
                m: m,
                n: batch,
                k: k,
                A: LibTorchNativeMethod.THSStorage_data_ptr(input.Handle),
                B: LibTorchNativeMethod.THSStorage_data_ptr(quantizedWeight.T.Handle),
                absmax: LibTorchNativeMethod.THSStorage_data_ptr(absMax.Handle),
                datatype: LibTorchNativeMethod.THSStorage_data_ptr(code.Handle),
                output: LibTorchNativeMethod.THSStorage_data_ptr(output.Handle),
                lda: lda,
                ldb: ldb,
                ldc: ldc,
                blocksize: blockSize,
                stream: IntPtr.Zero);
        }
        else if (input.dtype == ScalarType.BFloat16)
        {
            BitsAndBytesCudaNative.cgemm_4bit_inference_naive_bf16(
                m: m,
                n: batch,
                k: k,
                A: LibTorchNativeMethod.THSStorage_data_ptr(input.Handle),
                B: LibTorchNativeMethod.THSStorage_data_ptr(quantizedWeight.T.Handle),
                absmax: LibTorchNativeMethod.THSStorage_data_ptr(absMax.Handle),
                datatype: LibTorchNativeMethod.THSStorage_data_ptr(code.Handle),
                output: LibTorchNativeMethod.THSStorage_data_ptr(output.Handle),
                lda: lda,
                ldb: ldb,
                ldc: ldc,
                blocksize: blockSize,
                stream: IntPtr.Zero);
        }
        else
        {
            throw new NotImplementedException();
        }

        return output;
    }


    public static torch.Tensor CreateDynamicMap(bool signed = true, int maxExponentBits = 7, int totalBits = 8)
    {
        var data = new List<float>();
        int nonSignBits = totalBits - (signed ? 1 : 0);
        int additionalItems = (int)Math.Pow(2, nonSignBits - maxExponentBits) - 1;

        for (int i = 0; i < maxExponentBits; i++)
        {
            int fractionItems = signed
                ? (int)Math.Pow(2, i + nonSignBits - maxExponentBits) + 1
                : (int)Math.Pow(2, i + nonSignBits - maxExponentBits + 1) + 1;

            var boundaries = torch.linspace(0.1, 1, fractionItems);
            var means = (boundaries[..^1] + boundaries[1..]) / 2.0;
            data.AddRange((torch.pow(10f, i - (maxExponentBits - 1)) * means).data<float>().ToArray());

            if (signed)
            {
                data.AddRange((-(torch.pow(10f, (-(maxExponentBits - 1) + i)) * means)).data<float>().ToArray());
            }
        }

        if (additionalItems > 0)
        {
            var boundaries = torch.linspace(0.1, 1, additionalItems + 1);
            var means = (boundaries[..^1] + boundaries[1..]) / 2.0;
            data.AddRange((torch.pow(10f, -(maxExponentBits - 1) + maxExponentBits - 1) * means).data<float>().ToArray());

            if (signed)
            {
                data.AddRange((-(torch.pow(10f, -(maxExponentBits - 1) + maxExponentBits - 1) * means)).data<float>().ToArray());
            }
        }

        data.AddRange(new float[] { 0, 1.0f });

        if (data.Count != Math.Pow(2, totalBits))
        {
            int gap = 256 - data.Count;
            for (int i = 0; i < gap; i++)
            {
                data.Add(0);
            }
        }

        data.Sort();
        return torch.tensor(data.ToArray());
    }
}
