using BenchmarkDotNet.Running;
using TorchSharp.BitsAndBytes.Benchmark;
new CudaBenchmark().GEMM_INT8();
BenchmarkRunner.Run<CudaBenchmark>();