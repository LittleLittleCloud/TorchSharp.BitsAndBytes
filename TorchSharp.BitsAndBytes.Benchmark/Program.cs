using BenchmarkDotNet.Running;
using TorchSharp.BitsAndBytes.Benchmark;
BenchmarkRunner.Run<CudaBenchmark>();