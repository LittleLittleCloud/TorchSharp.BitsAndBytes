using System;
using System.Runtime.InteropServices;

namespace TorchSharp.BitsAndBytes; 
public static class LibTorchNativeMethod 
{ 
    [DllImport("LibTorchSharp")]
    public static extern IntPtr THSStorage_data_ptr(IntPtr tensor);
}

