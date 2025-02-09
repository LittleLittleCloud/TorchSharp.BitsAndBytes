using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;

namespace TorchSharp.BitsAndBytes.Benchmark;

public class CudaBenchmark : IDisposable
{
    private Tensor a1;
    private Tensor b;
    private string quantizedDType = "fp4";
    private int blockSize = 64;
    private int dim = 1024;

    public CudaBenchmark()
    {
        a1 = torch.rand(new long[] { dim * 4, dim }, dtype: ScalarType.Float32).cuda();
    }

    private torch.Tensor quantizedTensor;
    private torch.Tensor absMax;

    [GlobalSetup]
    public void Setup()
    {
        b = torch.rand(new long[] { 4 * dim, dim }, dtype: ScalarType.Float32).cuda();
        (quantizedTensor, absMax, _, _) = BitsAndByteUtils.Quantize4Bit(b, "fp4", blockSize);
    }

    [Benchmark]
    public void Quantize4Bit()
    {
        var result = BitsAndByteUtils.Quantize4Bit(a1, quantizedDType, blockSize);
    }

    [Benchmark]
    public void Dequantize4Bit()
    {
        var (quantizedTensor, absMax, _, n) = BitsAndByteUtils.Quantize4Bit(a1, quantizedDType, blockSize);
        var result = BitsAndByteUtils.Dequantize4Bit(quantizedTensor, absMax, ScalarType.Float32, quantizedDType, n, a1.shape, blockSize);
    }

    [Benchmark]
    public void GEMV_4Bit_FP4()
    {
        using var input = torch.rand(new long[] { 1, dim }, dtype: ScalarType.Float32).cuda();
        using var result = BitsAndByteUtils.Gemv4Bit(input, quantizedTensor, [4*dim, dim], absMax, blockSize, quantizedDType);
    }

    [Benchmark]
    public void GEMV_FP32()
    {
        using var input = torch.rand(new long[] { 1, dim }, dtype: ScalarType.Float32).cuda();
        using var result = torch.matmul(input, b.T);
    }

    public void Dispose()
    {
        a1.Dispose();
        b.Dispose();
    }
}
