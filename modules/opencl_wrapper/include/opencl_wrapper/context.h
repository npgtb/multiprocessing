#ifndef OPENCL_WRAPPER_CONTEXT_H
#define OPENCL_WRAPPER_CONTEXT_H

#include <memory>
#include <string>
#include <vector>
#include <opencl_wrapper/image.h>
#include <opencl_wrapper/opencl.h>
#include <opencl_wrapper/buffer.h>
#include <opencl_wrapper/program.h>
#include <opencl_wrapper/error_or.h>
#include <opencl_wrapper/command_queue.h>

namespace clw{
    class Device;
    class Context : public std::enable_shared_from_this<Context>{
        public:
            
            //Initialize empty context
            Context();

            //Initialize context with context_ptr
            Context(cl_context context_ptr);

            //Copy constructors
            Context(Context && other);
            Context(const Context&) = delete;

            //Free the opencl context memory
            ~Context();

            //Assignment operators
            Context& operator=(Context&& other);
            Context& operator=(const Context&) = delete;

            //Frees the memory tied to the cl object
            void release();

            //Creates a command queue for the context within the given device
            ErrorOr<std::shared_ptr<CommandQueue>> create_cmd_queue(const Device& device, const cl_queue_properties* properties);

            //Loads a program from the given file
            ErrorOr<std::shared_ptr<Program>> load_program_file(const std::string& source_file, std::vector<Device>& build_devices);

            //Creates a buffer object
            ErrorOr<Buffer> create_buffer(cl_mem_flags flags, const size_t buffer_size);

            //Creates a buffer object
            ErrorOr<Buffer> create_buffer(const BufferDescription& description);

            //Creates array of buffers based on the given descriptions
            cl_int create_buffers(std::vector<Buffer>& buffers, std::vector<BufferDescription>& descriptions);

            //Creates a image object
            ErrorOr<Image> create_image(
                cl_mem_flags flags, const size_t width, const size_t height, const size_t pitch, cl_channel_order format,
                cl_channel_type data_type, cl_mem_object_type image_type, void * host_data
            );

            //Creates a image object
            ErrorOr<Image> create_image(const ImageDescription& description);

            //Creates array of images based on the given descriptions
            cl_int create_images(std::vector<Image>& images, std::vector<ImageDescription>& descriptions);

        private:
            cl_context context_;
    };
}

#endif