namespace TorchSharp.BitsAndBytes.Tests;

public class CudaTheoryAttribute : TheoryAttribute
{
    public CudaTheoryAttribute()
    {
        if (Environment.GetEnvironmentVariable("DisableCudaTest") == "true")
        {
            Skip = "CUDA tests are disabled.";
        }
    }
}
