#ifndef OPENCL_WRAPPER_IMAGE_H
#define OPENCL_WRAPPER_IMAGE_H

#include <memory>
#include <opencl_wrapper/opencl.h>
#include <opencl_wrapper/error_or.h>

namespace clw{
    class Kernel;
    class Context;
    class CommandQueue;

    //Struct for descriping the features of the image
    struct ImageDescription{
        cl_mem_flags memory_flags;
        cl_image_format format;
        cl_image_desc description;
        void * host_data;
    };

    class Image{
        public:
            //Initialize empty image
            Image();

            //Initialize image
            Image(cl_mem image, std::shared_ptr<Context> context);

            //Copy constructors
            Image(Image && other);
            Image(const Image&) = delete;

            //Free buffer memory
            ~Image();

            //Assignment operators
            Image& operator=(Image&& other);
            Image& operator=(const Image&) = delete;

            //Frees the memory tied to the cl object
            void release();

            //Gets the buffer size
            ErrorOr<size_t> size();
            
        private:
            //Friend so we can bind it as argumentation and read its contents
            friend Kernel;
            friend CommandQueue;
            cl_mem image_;
            std::shared_ptr<Context> context_;
    };
}

#endif