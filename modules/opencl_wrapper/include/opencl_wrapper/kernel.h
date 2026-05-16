#ifndef OPENCL_WRAPPER_KERNEL_H
#define OPENCL_WRAPPER_KERNEL_H

#include <memory>
#include <opencl_wrapper/image.h>
#include <opencl_wrapper/local.h>
#include <opencl_wrapper/opencl.h>
#include <opencl_wrapper/buffer.h>

namespace clw{
    class Program;
    class CommandQueue;
    class Kernel{
        public:
            //Initialize empty Kernel
            Kernel();

            //Initialize Kernel with id
            Kernel(cl_kernel kernel, std::shared_ptr<Program> program);

            //Copy constructors
            Kernel(Kernel && other);
            Kernel(const Kernel&) = delete;

            //Free kernel memory
            ~Kernel();

            //Assignment operators
            Kernel& operator=(Kernel&& other);
            Kernel& operator=(const Kernel&) = delete;

            //Frees the memory tied to the cl object
            void release();

            //Bind a buffer to the kernel argumentation
            cl_int bind_arg(cl_uint arg_index, Buffer& bfr); 

            //Bind a image to the kernel argumentation
            cl_int bind_arg(cl_uint arg_index, Image& img); 

            //Bind a local to the kernel argumentation
            cl_int bind_arg(cl_uint arg_index, Local& bfr); 

            //Bind a fundmental type to the kernel argumentation
            template <typename T> requires std::is_fundamental_v<T>
            cl_int bind_arg(cl_uint arg_index, T arg, size_t size){
                return clSetKernelArg(kernel_, arg_index, size, &arg);
            }


        private:
            //Friend so we can queueu the execution of the kernel
            friend CommandQueue;
            cl_kernel kernel_;
            std::shared_ptr<Program> program_;
    };
}

#endif