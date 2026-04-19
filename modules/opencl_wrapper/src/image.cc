#include <opencl_wrapper/image.h>

namespace clw{
    //Initialize empty image
    Image::Image():image_(nullptr), context_(nullptr){}

    //Initialize image
    Image::Image(cl_mem image, std::shared_ptr<Context> context):image_(image), context_(context){

    }

    //Copy constructors
    Image::Image(Image && other):image_(other.image_), context_(other.context_){
        other.image_ = nullptr;
        other.context_ = nullptr;
    }

    //Free buffer memory
    Image::~Image(){
        release();
    }

    //Assignment operators
    Image& Image::operator=(Image&& other){
        image_ = other.image_;
        context_ = other.context_;
        other.image_ = nullptr;
        other.context_ = nullptr;
        return *this;
    }

    //Frees the memory tied to the cl object
    void Image::release(){
        if(image_){
            clReleaseMemObject(image_);
            image_ = nullptr;
            context_ = nullptr;
        }
    }

    //Gets the buffer size
    ErrorOr<size_t> Image::size(){
        size_t image_size = 0;
        cl_int error_code = clGetMemObjectInfo(image_, CL_MEM_SIZE, sizeof(size_t), &image_size, nullptr);
        if(error_code == CL_SUCCESS){
            return image_size;
        }
        return error_code;
    }

}