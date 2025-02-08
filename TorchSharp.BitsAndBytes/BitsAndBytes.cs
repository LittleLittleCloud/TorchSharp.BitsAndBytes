using System;
using System.Runtime.InteropServices;

namespace TorchSharp.BitsAndBytes;

/// <summary>
/// Provides access to bitsandbytes CUDA operations for quantized neural networks
/// </summary>
public static class BitsAndBytesCudaNative
{
    private const string DllName = "libbitsandbytes_cuda121";

    /// <summary>
    /// Represents the CUDA __nv_bfloat16 type
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct NvBFloat16
    {
        public ushort Value;
    }

    [DllImport(DllName)]
    public static extern void cdequantize_blockwise_fp32(
        IntPtr code,        // float*
        IntPtr A,          // float*
        IntPtr absmax,     // float*
        IntPtr output,     // unsigned char*
        int blocksize,
        int n,             // total size
        IntPtr stream);

    [DllImport(DllName)]
    public static extern void cdequantize_blockwise_fp16(
        IntPtr code,        // float*
        IntPtr A,          // float*
        IntPtr absmax,     // float*
        IntPtr output,     // unsigned char*
        int blocksize,
        int n,             // total size
        IntPtr stream);

    [DllImport(DllName)]
    public static extern void cdequantize_blockwise_bf16(
        IntPtr code,        // float*
        IntPtr A,          // float*
        IntPtr absmax,     // float*
        IntPtr output,     // unsigned char*
        int blocksize,
        int n,             // total size
        IntPtr stream);

    [DllImport(DllName)]
    public static extern void cdequantize_blockwise_fp32_fp4(
        IntPtr code,        // float*
        IntPtr A,          // float*
        IntPtr absmax,     // float*
        IntPtr output,     // unsigned char*
        int blocksize,
        int n,             // total size
        IntPtr stream);

    [DllImport(DllName)]
    public static extern void cdequantize_blockwise_fp32_nf4(
        IntPtr code,        // float*
        IntPtr A,          // float*
        IntPtr absmax,     // float*
        IntPtr output,     // unsigned char*
        int blocksize,
        int n,             // total size
        IntPtr stream);

    [DllImport(DllName)]
    public static extern void cdequantize_blockwise_fp16_fp4(
        IntPtr code,        // float*
        IntPtr A,          // float*
        IntPtr absmax,     // float*
        IntPtr output,     // unsigned char*
        int blocksize,
        int n,             // total size
        IntPtr stream);

    [DllImport(DllName)]
    public static extern void cdequantize_blockwise_fp16_nf4(
        IntPtr code,        // float*
        IntPtr A,          // float*
        IntPtr absmax,     // float*
        IntPtr output,     // unsigned char*
        int blocksize,
        int n,             // total size
        IntPtr stream);

    [DllImport(DllName)]
    public static extern void cdequantize_blockwise_bf16_fp4(
        IntPtr code,        // float*
        IntPtr A,          // float*
        IntPtr absmax,     // float*
        IntPtr output,     // unsigned char*
        int blocksize,
        int n,             // total size
        IntPtr stream);

    [DllImport(DllName)]
    public static extern void cdequantize_blockwise_bf16_nf4(
        IntPtr code,        // float*
        IntPtr A,          // float*
        IntPtr absmax,     // float*
        IntPtr output,     // unsigned char*
        int blocksize,
        int n,             // total size
        IntPtr stream);

    [DllImport(DllName)]
    public static extern void cquantize_blockwise_fp32_fp4(
        IntPtr code,        // float*
        IntPtr A,          // float*
        IntPtr absmax,     // float*
        IntPtr output,     // unsigned char*
        int blocksize,
        int n             // total size
     );

    [DllImport(DllName)]
    public static extern void cquantize_blockwise_fp32_nf4(
        IntPtr code,        // float*
        IntPtr A,          // float*
        IntPtr absmax,     // float*
        IntPtr output,     // unsigned char*
        int blocksize,
        int n             // total size
     );

    [DllImport(DllName)]
    public static extern void cquantize_blockwise_fp32(
        IntPtr code,        // float*
        IntPtr A,          // float*
        IntPtr absmax,     // float*
        IntPtr output,     // unsigned char*
        int blocksize,
        int n             // total size
     );

    [DllImport(DllName)]
    public static extern void cquantize_blockwise_fp16_fp4(
        IntPtr code,        // float*
        IntPtr A,          // float*
        IntPtr absmax,     // float*
        IntPtr output,     // unsigned char*
        int blocksize,
        int n             // total size
     );

    [DllImport(DllName)]
    public static extern void cquantize_blockwise_fp16_nf4(
        IntPtr code,        // float*
        IntPtr A,          // float*
        IntPtr absmax,     // float*
        IntPtr output,     // unsigned char*
        int blocksize,
        int n             // total size
     );

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void cquantize_blockwise_bf16_fp4(
        IntPtr code,        // float*
        IntPtr A,          // __nv_bfloat16*
        IntPtr absmax,     // float*
        IntPtr output,     // unsigned char*
        int blocksize,
        int n             // total size
    );

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void cquantize_blockwise_bf16_nf4(
        IntPtr code,        // float*
        IntPtr A,          // __nv_bfloat16*
        IntPtr absmax,     // float*
        IntPtr output,     // unsigned char*
        int blocksize,
        int n             // total size
    );

    [DllImport(DllName)]
    public static extern void cgemm_4bit_inference_naive_fp16(
        int m,
        int n,
        int k,
        IntPtr A,          // half*
        IntPtr B,          // unsigned char*
        IntPtr absmax,     // float*
        IntPtr datatype,   // float*
        IntPtr output,     // half*
        int lda,
        int ldb,
        int ldc,
        int blocksize,
        IntPtr stream      // cudaStream_t
    );

    [DllImport(DllName)]
    public static extern void cgemm_4bit_inference_naive_fp32(
        int m,
        int n,
        int k,
        IntPtr A,          // half*
        IntPtr B,          // unsigned char*
        IntPtr absmax,     // float*
        IntPtr datatype,   // float*
        IntPtr output,     // half*
        int lda,
        int ldb,
        int ldc,
        int blocksize,
        IntPtr stream      // cudaStream_t
    );

    [DllImport(DllName)]
    public static extern void cgemm_4bit_inference_naive_bf16(
        int m,
        int n,
        int k,
        IntPtr A,          // half*
        IntPtr B,          // unsigned char*
        IntPtr absmax,     // float*
        IntPtr datatype,   // float*
        IntPtr output,     // half*
        int lda,
        int ldb,
        int ldc,
        int blocksize,
        IntPtr stream      // cudaStream_t
    );

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void dequantize(
        IntPtr output,  // float* 
        IntPtr input,   // byte*
        IntPtr scale,   // float*
        int size,
        IntPtr stream   // cudaStream_t
    );
}