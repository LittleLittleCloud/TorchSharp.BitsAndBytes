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

    [CudaTheory]
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

    [CudaTheory]
    [InlineData(32, 1, false, false, 16)]
    [InlineData(32, 1, false, true, 16)]
    [InlineData(32, 1, true, false, 16)]
    [InlineData(32, 1, true, true, 16)]
    [InlineData(64, 1, true, true, 16)]
    [InlineData(128, 1, true, true, 16)]
    [InlineData(512, 1, true, true, 16)]
    [InlineData(32, 1, true, true, 512)]
    [InlineData(32, 16, false, false, 16)]
    [InlineData(32, 16, false, true, 16)]
    [InlineData(32, 8, true, false, 16)]
    [InlineData(32, 4, true, true, 16)]
    [InlineData(128, 32, true, true, 16)]
    [InlineData(512, 32, true, true, 16)]
    [InlineData(32, 4, true, true, 512)]
    public void TestInt8GEMM(int hiddenDim, int batchDim, bool transposeInput, bool transposeWeight, int seqDim)
    {
        // 2-D input
        foreach (int i in Enumerable.Range(0, 20))
        {
            long[] inputShape = !transposeInput ? [batchDim, hiddenDim] : [hiddenDim, batchDim];
            var outputChannel = 32 * new Random().Next(1, 10);
            long[] weightShape = transposeWeight ? [outputChannel, hiddenDim] : [hiddenDim, outputChannel];

            using var input = torch.randint(-128, 127, inputShape, ScalarType.Int8).cuda();
            using var weight = torch.randint(-128, 127, weightShape, ScalarType.Int8).cuda();
            using var baseline = (transposeInput, transposeWeight) switch
            {
                (false, false) => torch.matmul(input.to_type(ScalarType.Float32), weight.to_type(ScalarType.Float32)),
                (false, true) => torch.matmul(input.to_type(ScalarType.Float32), weight.to_type(ScalarType.Float32).t()),
                (true, false) => torch.matmul(input.to_type(ScalarType.Float32).t(), weight.to_type(ScalarType.Float32)),
                (true, true) => torch.matmul(input.to_type(ScalarType.Float32).t(), weight.to_type(ScalarType.Float32).t()),
            };
            using var result = (transposeInput, transposeWeight) switch
            {
                (false, false) => Function.Int8GEMM(input, weight),
                (false, true) => Function.Int8GEMM(input, weight.t()),
                (true, false) => Function.Int8GEMM(input.t(), weight),
                (true, true) => Function.Int8GEMM(input.t(), weight.t()),
            };

            var diff = baseline - result.to_type(ScalarType.Float32);
            var avg = diff.abs().mean().data<float>();

            Assert.True(avg[0] <= 1e-5);
        }

        // 3-dim input
        foreach (int i in Enumerable.Range(0, 20))
        {
            if (transposeInput)
            {
                // skip 3-dim input with transposeInput = true
                continue;
            }
            long[] inputShape = [batchDim, seqDim, hiddenDim];
            var outputChannel = 32 * new Random().Next(1, 10);
            long[] weightShape = transposeWeight ? [outputChannel, hiddenDim] : [hiddenDim, outputChannel];

            using var input = torch.randint(-128, 127, inputShape, ScalarType.Int8).cuda();
            using var weight = torch.randint(-128, 127, weightShape, ScalarType.Int8).cuda();
            using var baseline = (transposeInput, transposeWeight) switch
            {
                (false, false) => torch.matmul(input.to_type(ScalarType.Float32), weight.to_type(ScalarType.Float32)),
                (false, true) => torch.matmul(input.to_type(ScalarType.Float32), weight.to_type(ScalarType.Float32).t()),
                _ => throw new NotImplementedException()
            };
            using var result = (transposeInput, transposeWeight) switch
            {
                (false, false) => Function.Int8GEMM(input, weight),
                (false, true) => Function.Int8GEMM(input, weight.t()),
                _ => throw new NotImplementedException()
            };

            var diff = baseline - result.to_type(ScalarType.Float32);
            var avg = diff.abs().mean().data<float>();

            Assert.True(avg[0] <= 1e-5);
        }
    }
    
    [CudaTheory]
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

    [CudaTheory]
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

    [CudaTheory]
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

        Tensor outTensor = null;
        sw.Start();
        for (int i = 0; i < 1000; i++)
        {
            outTensor = BitsAndByteUtils.Gemv4Bit(a, quantizedTensor.T, b.shape, absMax, blockSize, quantizedDType);
        }
        sw.Stop();
        output.WriteLine($"{dtype}-{quantizedDType}-{dim} Time taken for 4-bit GEMV: {sw.ElapsedMilliseconds} ms");
        Tensor outBaseline = null;
        sw.Restart();
        for(int i = 0; i < 1000; i++)
        {
            outBaseline = torch.matmul(a, b.t());
        }
        sw.Stop();

        output.WriteLine($"{dtype}-{quantizedDType}-{dim} Time taken for baseline GEMV: {sw.ElapsedMilliseconds} ms");
        var abs = outBaseline - outTensor;
        var avg = abs.to(ScalarType.Float32).abs().mean().data<float>();
        Assert.Equal(1, avg.Count);
        Assert.True(avg.First() == 0);
    }

    [Fact]
    public void TestCheckMatmul_ValidInputs()
    {
        var A = torch.randint(0, 10, new long[] { 2, 3 }, ScalarType.Int8);
        var B = torch.randint(0, 10, new long[] { 3, 2 }, ScalarType.Int8);

        var result = BitsAndByteUtils.CheckMatmul(A, B, false, false, ScalarType.Int8);

        Assert.Equal([2, 2], result);
    }

    [Fact]
    public void TestCheckMatmul_InvalidInputs()
    {
        var A = torch.randint(0, 10, new long[] { 2, 3 }, ScalarType.Int8);
        var B = torch.randint(0, 10, new long[] { 2, 2 }, ScalarType.Int8);

        Assert.Throws<ArgumentException>(() => BitsAndByteUtils.CheckMatmul(A, B, false, false, ScalarType.Int8));
    }

    [Fact]
    public void TestCheckMatmul_TransposedInputs()
    {
        var A = torch.randint(0, 10, new long[] { 3, 2 }, ScalarType.Int8);
        var B = torch.randint(0, 10, new long[] { 3, 2 }, ScalarType.Int8);

        var result = BitsAndByteUtils.CheckMatmul(A, B, true, false, ScalarType.Int8);

        Assert.Equal([2, 2], result);
    }

    [Fact]
    public void TestCheckMatmul_NullOutput()
    {
        var A = torch.randint(0, 10, new long[] { 2, 3 }, ScalarType.Int8);
        var B = torch.randint(0, 10, new long[] { 3, 2 }, ScalarType.Int8);

        var result = BitsAndByteUtils.CheckMatmul(A, B, false, false, ScalarType.Int8);

        Assert.Equal([2, 2], result);
    }
}