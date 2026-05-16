#include <opencl_wrapper/kernel.h>

namespace clw{
    //Initialize empty Kernel
    Kernel::Kernel():program_(nullptr), kernel_(nullptr){}

    //Initialize Kernel with id
    Kernel::Kernel(cl_kernel kernel, std::shared_ptr<Program> program):kernel_(kernel), program_(program){}

    //Copy constructors
    Kernel::Kernel(Kernel && other):kernel_(other.kernel_), program_(other.program_){
        other.kernel_ = nullptr;
        other.program_ = nullptr;
    }

    //Free kernel memory
    Kernel::~Kernel(){
        release();
    }

    //Assignment operators
    Kernel& Kernel::operator=(Kernel&& other){
        kernel_ = other.kernel_;
        program_ = other.program_;
        other.kernel_ = nullptr;
        other.program_ = nullptr;
        return *this;
    }

    //Frees the memory tied to the cl object
    void Kernel::release(){
        if(kernel_){
            clReleaseKernel(kernel_);
            kernel_ = nullptr;
            program_ = nullptr;
        }
    }

    //Bind a buffer to the kernel argumentation
    cl_int Kernel::bind_arg(cl_uint arg_index, Buffer& bfr){
        return clSetKernelArg(kernel_, arg_index, sizeof(cl_mem), &bfr.buffer_);
    }

    //Bind a image to the kernel argumentation
    cl_int Kernel::bind_arg(cl_uint arg_index, Image& img){
        return clSetKernelArg(kernel_, arg_index, sizeof(cl_mem), &img.image_);
    }

    //Bind a local to the kernel argumentation
    cl_int Kernel::bind_arg(cl_uint arg_index, Local& bfr){
        return clSetKernelArg(kernel_, arg_index, bfr.size, nullptr);
    }

}