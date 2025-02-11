using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;

namespace TorchSharp.BitsAndBytes;

public class Function
{
    /// <summary>
    /// Integer General Matrix Multiplication (IGEMM) for 8-bit integer data types.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="weight"></param>
    /// <param name="transposeWeight"></param>
    /// <param name="transposeInput"></param>
    /// <returns></returns>
    public static Tensor Int8GEMM(
        Tensor input,
        Tensor weight,
        bool transposeWeight = false,
        bool transposeInput = false)
    {
        var sout = BitsAndByteUtils.CheckMatmul(input, weight, transposeWeight, transposeInput);
        var @out = torch.zeros((long[])[.. sout], dtype: torch.int32, device: input.device);
        if (input.shape.Length == 3 && weight.shape.Length == 3)
        {
            if (input.shape[0] == weight.shape[0] && input.shape[2] == weight.shape[1])
            {
                throw new NotImplementedException();
            }
        }

        var inputShape = input.IntShape().ToArray();
        var weightShape = weight.IntShape().ToArray();
        if (transposeInput && inputShape.Length == 2)
        {
            inputShape = [inputShape[1], inputShape[0]];
        }
        else if (transposeInput && inputShape.Length == 3)
        {
            inputShape = [inputShape[0], inputShape[2], inputShape[0]];
        }
        if (transposeWeight && weightShape.Length == 2)
        {
            weightShape = [weightShape[1], weightShape[0]];
        }
        else if (transposeWeight && weightShape.Length == 3)
        {
            weightShape = [weightShape[0], weightShape[2], weightShape[0]];
        }
        // this is a mess: cuBLAS expect column major, but PyTorch is row major.
        // So to perform the matrix multiplication, we have to treat A, B, and C matrices
        // (transpose of row major is column major)
        // This means we compute B^T A^T = C^T and we explicitly switch the dimensions of each of these

        // matrices in the input arguments for cuBLAS
        // column major: A @ B = C: [m, k] @ [k, n] = [m, n]
        // row major: B^T @ A^T = C^T: [m, k] @ [k, n] = [m, n]
        // column major with row major layout: B^T @ A^T = C^T: [k, m] @ [n, k] = [n, m]
        int m = 0, n = 0, k = 0, lda = 0, ldb = 0, ldc = 0;

        if (weightShape.Length == 2)
        {
            if (weight.stride(0) == weight.shape[1])
            {
                transposeWeight = false;
            }
            else if (weight.stride(1) == weight.shape[0])
            {
                transposeWeight = true;
            }
            if (input.shape.Length == 2)
            {
                if (input.stride(0) == input.shape[1])
                {
                    transposeInput = false;
                }
                else if (input.stride(1) == input.shape[0])
                {
                    transposeInput = true;
                }
            }
            else
            {
                if (input.stride(1) == input.shape[2])
                {
                    transposeInput = false;
                }
                else if (input.stride(2) == input.shape[1])
                {
                    transposeInput = true;
                }
            }

            if (inputShape.Length == 2)
            {
                n = inputShape[0];
                ldb = (int)input.stride(transposeInput ? 1 : 0);
            }
            else if (inputShape.Length == 3 && weightShape.Length == 2)
            {
                n = inputShape[0] * inputShape[1];
                ldb = inputShape[2];
            }

            m = weightShape[1];
            k = weightShape[0];
            lda = (int)weight.stride(transposeWeight ? 1 : 0);
            ldc = weightShape[1];
        }
        else if (weightShape.Length == 3)
        {
            // special case
            if (!(inputShape[0] == weightShape[0] && inputShape[1] == weightShape[1]))
            {
                throw new ArgumentException($"Only bsi,bso->io supported for tensor contractions, but dims for A x B were: {inputShape} x {weightShape}");
            }

            transposeInput = true;
            transposeWeight = false;
            m = weightShape[2];
            n = inputShape[2];
            k = weightShape[0] * weightShape[1];

            lda = m;
            ldb = inputShape[2];
            ldc = m;
        }

        var context = BitsAndBytesCudaNative.get_context();
        var A = LibTorchNativeMethod.THSStorage_data_ptr(input.Handle);
        var B = LibTorchNativeMethod.THSStorage_data_ptr(weight.Handle);
        var C = LibTorchNativeMethod.THSStorage_data_ptr(@out.Handle);

        BitsAndBytesCudaNative.cigemm(
            context: context,
            transposeA: transposeWeight, // cuBLAS expects column major, but PyTorch is row major
            transposeB: transposeInput, // So we have to transpose A and B
            m: m,   
            n: n,
            k: k,
            A: B,   // out_T = B_T @ A_T
            B: A,
            C: C,
            lda: lda,
            ldb: ldb,
            ldc: ldc);

        return @out;
    }
}
