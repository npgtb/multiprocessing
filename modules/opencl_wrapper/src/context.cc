#include <fstream>
#include <memory>
#include <sstream>
#include <opencl_wrapper/device.h>
#include <opencl_wrapper/context.h>

namespace clw{
    //Initialize empty context
    Context::Context():context_(nullptr){}

    //Initialize context with context_ptr
    Context::Context(cl_context context_ptr):context_(context_ptr){}

    //Copy constructors
    Context::Context(Context && other):context_(other.context_){
        other.context_ = nullptr;
    }

    //Free the opencl context memory
    Context::~Context(){
        release();
    }

    //Assignment operators
    Context& Context::operator=(Context&& other){
        context_ = other.context_;
        other.context_ = nullptr;
        return *this;
    }

    //Frees the memory tied to the cl object
    void Context::release(){
        if(context_){
            clReleaseContext(context_);
            context_ = nullptr;
        }
    }

    //Creates a command queue for the context within the given device
    ErrorOr<std::shared_ptr<CommandQueue>> Context::create_cmd_queue(const Device& device, const cl_queue_properties* properties){
        cl_int error_code = 0;
        //https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clCreateCommandQueueWithProperties.html
        //creates a cmd queue for the context within the device: context, device, behaviour properties, error_code
        std::shared_ptr<CommandQueue> queue = std::make_shared<CommandQueue>(
            clCreateCommandQueueWithProperties(context_, device.id, properties, &error_code),
            shared_from_this()
        );
        if(error_code == CL_SUCCESS){
            return queue;
        }
        return error_code;
    }

    //Reads a file into a string 
    std::string read_file(const std::string& path){
        std::ifstream source_file(path);
        if(source_file.is_open()){
            std::stringstream contents;
            contents << source_file.rdbuf();
            return contents.str();
        }
        return "";
    }

    //Loads a program from the given file
    ErrorOr<std::shared_ptr<Program>> Context::load_program_file(const std::string& source_file){
        cl_int error_code = CL_INVALID_VALUE;
        std::string source = read_file(source_file);
        if(!source.empty()){
            //null terminate the string, clCreateProgramWithSource expects a null terminated
            //when src_lenghts is null
            source += '\0';
            const char * source_codes = source.c_str();
            //https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clCreateProgramWithSource.html
            //creates a holding obj for the program source: context, src_count, source, src_lengths, error_code
            cl_program program_id = clCreateProgramWithSource(context_, 1, &source_codes, NULL, &error_code);
            if(error_code != CL_SUCCESS){
                return error_code;
            }
            return std::make_shared<Program>(program_id, shared_from_this());
        }
        return error_code;
    }

    //Creates a buffer object
    ErrorOr<Buffer> Context::create_buffer(cl_mem_flags flags, const size_t buffer_size){
        cl_int error_code = 0;
        //https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clCreateBuffer.html
        //creates a memory object: context, memory flags, buffer size, ptr to existing buffer, error code
        cl_mem memory = clCreateBuffer(context_, flags, buffer_size, NULL, &error_code);
        if(error_code == CL_SUCCESS){
            return Buffer(memory, shared_from_this());
        }
        return error_code;
    }

    //Creates a buffer object
    ErrorOr<Buffer> Context::create_buffer(const BufferDescription& description){
        return create_buffer(description.memory_flags, description.buffer_size);
    }

    //Creates array of buffers based on the given descriptions
    cl_int Context::create_buffers(std::vector<Buffer>& buffers, std::vector<BufferDescription>& descriptions){
        for(const auto& description: descriptions){
            ErrorOr<Buffer> buffer_creation = create_buffer(description);
            if(!buffer_creation.ok()){
                return buffer_creation.error();
            }
            buffers.push_back(std::move(buffer_creation).value());
        }
        return CL_SUCCESS;
    }

    //Creates a image object
    ErrorOr<Image> Context::create_image(
        cl_mem_flags flags, const size_t width, const size_t height, const size_t pitch, cl_channel_order format,
        cl_channel_type data_type, cl_mem_object_type image_type, void * host_data
    ){
        cl_int error_code = 0;
        cl_image_format image_format = {.image_channel_order = format, .image_channel_data_type = data_type};
        cl_image_desc image_description = {.image_type = image_type, .image_width = width, .image_height = height, .image_row_pitch = pitch};
        //https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clCreateImageWithProperties.html
        //creates a image object: context, mem flags, image format struct, image desc struct, host_ptr, error_code 
        cl_mem memory = clCreateImage(
            context_, flags, &image_format, &image_description,
            host_data, &error_code
        );
        if(error_code == CL_SUCCESS){
            return Image(memory, shared_from_this());
        }
        return error_code;
    }

    //Creates a image object
    ErrorOr<Image> Context::create_image(const ImageDescription& description){
        cl_int error_code = 0;
        //https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clCreateImageWithProperties.html
        //creates a image object: context, mem flags, image format struct, image desc struct, host_ptr, error_code 
        cl_mem memory = clCreateImage(
            context_, description.memory_flags, &description.format, &description.description,
            description.host_data, &error_code
        );
        if(error_code == CL_SUCCESS){
            return Image(memory, shared_from_this());
        }
        return error_code;
    }

    //Creates array of images based on the given descriptions
    cl_int Context::create_images(std::vector<Image>& images, std::vector<ImageDescription>& descriptions){
        for(const auto& description: descriptions){
            ErrorOr<Image> image_creation = create_image(description);
            if(!image_creation.ok()){
                return image_creation.error();
            }
            images.push_back(std::move(image_creation).value());
        }
        return CL_SUCCESS;
    }

}