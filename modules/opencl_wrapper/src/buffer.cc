#include <opencl_wrapper/buffer.h>

namespace clw{
    //Initialize empty buffer
    Buffer::Buffer():buffer_(nullptr), context_(nullptr){}

    //Initialize memory buffer
    Buffer::Buffer(cl_mem buffer, std::shared_ptr<Context> context):buffer_(buffer), context_(context){}

    //Copy constructors
    Buffer::Buffer(Buffer && other):buffer_(other.buffer_), context_(other.context_){
        other.buffer_ = nullptr;
        other.context_ = nullptr;
    }

    //Free buffer memory
    Buffer::~Buffer(){
        release();
    }

    //Assignment operators
    Buffer& Buffer::operator=(Buffer&& other){
        buffer_ = other.buffer_;
        context_ = other.context_;
        other.buffer_ = nullptr;
        other.context_ = nullptr;
        return *this;
    }

    //Frees the memory tied to the cl object
    void Buffer::release(){
        if(buffer_){
            clReleaseMemObject(buffer_);
            buffer_ = nullptr;
            context_ = nullptr;
        }
    }

    //Gets the buffer size
    ErrorOr<size_t> Buffer::size(){
        size_t buffer_size = 0;
        cl_int error_code = clGetMemObjectInfo(buffer_, CL_MEM_SIZE, sizeof(size_t), &buffer_size, nullptr);
        if(error_code == CL_SUCCESS){
            return buffer_size;
        }
        return error_code;
    }


}