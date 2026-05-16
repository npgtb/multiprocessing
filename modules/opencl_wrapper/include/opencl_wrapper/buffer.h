#ifndef OPENCL_WRAPPER_BUFFER_H
#define OPENCL_WRAPPER_BUFFER_H

#include <memory>
#include <opencl_wrapper/opencl.h>
#include <opencl_wrapper/error_or.h>

namespace clw{
    class Kernel;
    class Context;
    class CommandQueue;

    //Struct for descriping the features of the buffer
    struct BufferDescription{
        cl_mem_flags memory_flags;
        size_t buffer_size;
    };

    class Buffer{
        public:
            //Initialize empty buffer
            Buffer();

            //Initialize memory buffer
            Buffer(cl_mem buffer, std::shared_ptr<Context> context);

            //Copy constructors
            Buffer(Buffer && other);
            Buffer(const Buffer&) = delete;

            //Free buffer memory
            ~Buffer();

            //Assignment operators
            Buffer& operator=(Buffer&& other);
            Buffer& operator=(const Buffer&) = delete;

            //Frees the memory tied to the cl object
            void release();

            //Gets the buffer size
            ErrorOr<size_t> size();
            
        private:
            //Friend so we can bind it as argumentation and read its contents
            friend Kernel;
            friend CommandQueue;
            cl_mem buffer_;
            std::shared_ptr<Context> context_;
    };
}

#endif