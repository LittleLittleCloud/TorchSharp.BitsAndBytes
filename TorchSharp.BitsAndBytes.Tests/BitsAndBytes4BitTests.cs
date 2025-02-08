using System.Diagnostics;
using System.Runtime.InteropServices;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace TorchSharp.BitsAndBytes.Tests;

public class BitsAndBytes4BitTests
{
    private readonly ITestOutputHelper output;
    public BitsAndBytes4BitTests(ITestOutputHelper output)
    {
        this.output = output;
    }

    [Theory]
    [InlineData(ScalarType.BFloat16, "fp4", 64)]
    [InlineData(ScalarType.BFloat16, "nf4", 64)]
    [InlineData(ScalarType.BFloat16, "fp4", 128)]
    [InlineData(ScalarType.BFloat16, "nf4", 128)]
    [InlineData(ScalarType.BFloat16, "fp4", 256)]
    [InlineData(ScalarType.BFloat16, "nf4", 256)]
    [InlineData(ScalarType.BFloat16, "fp4", 512)]
    [InlineData(ScalarType.BFloat16, "nf4", 512)]
    [InlineData(ScalarType.BFloat16, "fp4", 1024)]
    [InlineData(ScalarType.BFloat16, "nf4", 1024)]
    [InlineData(ScalarType.BFloat16, "fp4", 2048)]
    [InlineData(ScalarType.BFloat16, "nf4", 2048)]
    [InlineData(ScalarType.BFloat16, "fp4", 4096)]
    [InlineData(ScalarType.BFloat16, "nf4", 4096)]
    [InlineData(ScalarType.Float32, "fp4", 64)]
    [InlineData(ScalarType.Float32, "nf4", 64)]
    [InlineData(ScalarType.Float32, "fp4", 128)]
    [InlineData(ScalarType.Float32, "nf4", 128)]
    [InlineData(ScalarType.Float32, "fp4", 256)]
    [InlineData(ScalarType.Float32, "nf4", 256)]
    [InlineData(ScalarType.Float32, "fp4", 512)]
    [InlineData(ScalarType.Float32, "nf4", 512)]
    [InlineData(ScalarType.Float32, "fp4", 1024)]
    [InlineData(ScalarType.Float32, "nf4", 1024)]
    [InlineData(ScalarType.Float32, "fp4", 2048)]
    [InlineData(ScalarType.Float32, "nf4", 2048)]
    [InlineData(ScalarType.Float32, "fp4", 4096)]
    [InlineData(ScalarType.Float32, "nf4", 4096)]
    [InlineData(ScalarType.Float16, "fp4", 64)]
    [InlineData(ScalarType.Float16, "nf4", 64)]
    [InlineData(ScalarType.Float16, "fp4", 128)]
    [InlineData(ScalarType.Float16, "nf4", 128)]
    [InlineData(ScalarType.Float16, "fp4", 256)]
    [InlineData(ScalarType.Float16, "nf4", 256)]
    [InlineData(ScalarType.Float16, "fp4", 512)]
    [InlineData(ScalarType.Float16, "nf4", 512)]
    [InlineData(ScalarType.Float16, "fp4", 1024)]
    [InlineData(ScalarType.Float16, "nf4", 1024)]
    [InlineData(ScalarType.Float16, "fp4", 2048)]
    [InlineData(ScalarType.Float16, "nf4", 2048)]
    [InlineData(ScalarType.Float16, "fp4", 4096)]
    [InlineData(ScalarType.Float16, "nf4", 4096)]
    public void Test4BitQuant(ScalarType inputDType, string quantizedDType, int blockSize)
    {
        var a1 = torch.rand([1024 * 4, 1024], dtype: inputDType).cuda();
        (var quantiedTensor, var absMax, var _, var n) = BitsAndByteUtils.Quantize4Bit(a1, quantizedDType, blockSize);
        var dequantizedTensor = BitsAndByteUtils.Dequantize4Bit(quantiedTensor, absMax, inputDType, quantizedDType, n, a1.shape, blockSize);

        // Check that the dequantized tensor is close to the original tensor
        var abs = a1 - dequantizedTensor;
        var avg = abs.to(ScalarType.Float32).abs().mean().data<float>();
        Assert.Equal(1, avg.Count);
        Assert.True(avg.First() <= 0.2);
    }

    [Theory]
    [InlineData(ScalarType.Float32, "fp4", 64, 1024)]
    [InlineData(ScalarType.Float32, "nf4", 64, 1024)]
    [InlineData(ScalarType.Float16, "fp4", 64, 1024)]
    [InlineData(ScalarType.Float16, "nf4", 64, 1024)]
    [InlineData(ScalarType.BFloat16, "fp4", 64, 1024)]
    [InlineData(ScalarType.BFloat16, "nf4", 64, 1024)]
    public void TestGemv4Bit(ScalarType dtype, string quantizedDType, int blockSize, int dim)
    {
        var sw = new Stopwatch();
        var code  = BitsAndByteUtils.Get4BitType(quantizedDType, "cuda");
        var a = torch.rand([1, dim], dtype: dtype).cuda(); // input
        var b = torch.rand([dim * 4, dim], dtype: dtype).cuda(); // weight

        // quantize b
        (var quantizedTensor, var absMax, var _, var n) = BitsAndByteUtils.Quantize4Bit(b, quantizedDType, blockSize);
        
        sw.Start();
        var outTensor = BitsAndByteUtils.Gemv4Bit(a, quantizedTensor, b.shape, absMax, blockSize, quantizedDType);
        sw.Stop();
        output.WriteLine($"{dtype}-{quantizedDType}-{dim} Time taken for 4-bit GEMV: {sw.ElapsedMilliseconds} ms");
        sw.Reset();
        sw.Start();
        var outBaseline = torch.mm(a, b.t());
        sw.Stop();

        output.WriteLine($"{dtype}-{quantizedDType}-{dim} Time taken for baseline GEMV: {sw.ElapsedMilliseconds} ms");
        var abs = outBaseline - outTensor;
        var avg = abs.to(ScalarType.Float32).abs().mean().data<float>();
        Assert.Equal(1, avg.Count);
        Assert.True(avg.First() <= 1);
    }

    [Theory]
    [InlineData(ScalarType.Float32, "fp4", 128, 128)]
    [InlineData(ScalarType.Float32, "nf4", 128, 128)]
    [InlineData(ScalarType.Float16, "fp4", 128, 128)]
    [InlineData(ScalarType.Float16, "nf4", 128, 128)]
    [InlineData(ScalarType.BFloat16, "fp4", 128, 128)]
    [InlineData(ScalarType.BFloat16, "nf4", 128, 128)]
    public void TestGemv4Bit2D128(ScalarType dtype, string quantizedDType, int blockSize, int dim)
    {
        var sw = new Stopwatch();
        var code = BitsAndByteUtils.Get4BitType(quantizedDType, "cuda");
        var a = torch.ones([1, dim], dtype: dtype).cuda(); // input
        var b = torch.ones([dim * 4, dim], dtype: dtype).cuda(); // weight

        // quantize b
        (var quantizedTensor, var absMax, var _, var n) = BitsAndByteUtils.Quantize4Bit(b, quantizedDType, blockSize);

        sw.Start();
        var outTensor = BitsAndByteUtils.Gemv4Bit(a, quantizedTensor.T, b.shape, absMax, blockSize, quantizedDType);
        sw.Stop();
        output.WriteLine($"{dtype}-{quantizedDType}-{dim} Time taken for 4-bit GEMV: {sw.ElapsedMilliseconds} ms");
        sw.Reset();
        sw.Start();
        var outBaseline = torch.mm(a, b.t());
        sw.Stop();

        output.WriteLine($"{dtype}-{quantizedDType}-{dim} Time taken for baseline GEMV: {sw.ElapsedMilliseconds} ms");
        var abs = outBaseline - outTensor;
        var avg = abs.to(ScalarType.Float32).abs().mean().data<float>();
        Assert.Equal(1, avg.Count);
        Assert.True(avg.First() == 0);
    }

    [Theory]
    [InlineData(ScalarType.Float32, "fp4", 128, 128)]
    [InlineData(ScalarType.Float32, "nf4", 128, 128)]
    [InlineData(ScalarType.Float16, "fp4", 128, 128)]
    [InlineData(ScalarType.Float16, "nf4", 128, 128)]
    [InlineData(ScalarType.BFloat16, "fp4", 128, 128)]
    [InlineData(ScalarType.BFloat16, "nf4", 128, 128)]
    public void TestGemv4Bit3D128(ScalarType dtype, string quantizedDType, int blockSize, int dim)
    {
        var sw = new Stopwatch();
        var code = BitsAndByteUtils.Get4BitType(quantizedDType, "cuda");
        var a = torch.ones([1, 1, dim], dtype: dtype).cuda(); // input
        var b = torch.ones([dim * 4, dim], dtype: dtype).cuda(); // weight

        // quantize b
        (var quantizedTensor, var absMax, var _, var n) = BitsAndByteUtils.Quantize4Bit(b, quantizedDType, blockSize);

        sw.Start();
        var outTensor = BitsAndByteUtils.Gemv4Bit(a, quantizedTensor.T, b.shape, absMax, blockSize, quantizedDType);
        sw.Stop();
        output.WriteLine($"{dtype}-{quantizedDType}-{dim} Time taken for 4-bit GEMV: {sw.ElapsedMilliseconds} ms");
        sw.Reset();
        sw.Start();
        var outBaseline = torch.matmul(a, b.t());
        sw.Stop();

        output.WriteLine($"{dtype}-{quantizedDType}-{dim} Time taken for baseline GEMV: {sw.ElapsedMilliseconds} ms");
        var abs = outBaseline - outTensor;
        var avg = abs.to(ScalarType.Float32).abs().mean().data<float>();
        Assert.Equal(1, avg.Count);
        Assert.True(avg.First() == 0);
    }

}