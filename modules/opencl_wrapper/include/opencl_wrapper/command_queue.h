#ifndef OPENCL_WRAPPER_COMMAND_QUEUE_H
#define OPENCL_WRAPPER_COMMAND_QUEUE_H

#include <memory>
#include <opencl_wrapper/event.h>
#include <opencl_wrapper/kernel.h>
#include <opencl_wrapper/opencl.h>

namespace clw{

    class Context;
    class CommandQueue : public std::enable_shared_from_this<CommandQueue>{
        public:
            //Initialize empty CommandQueue
            CommandQueue();

            //Initialize CommandQueue with id
            CommandQueue(cl_command_queue queue, std::shared_ptr<Context> queue_context);

            //Copy constructors
            CommandQueue(CommandQueue && other);
            CommandQueue(const CommandQueue&) = delete;

            //Release the memory
            ~CommandQueue();

            //Assignment operators
            CommandQueue& operator=(CommandQueue&& other);
            CommandQueue& operator=(const CommandQueue&) = delete;

            //CCQueue kinda exists outside the destruction hierarchy, so manually release it
            void release();

            //Queues the execution of the kernel
            ErrorOr<std::shared_ptr<Event>> execute_kernel(std::shared_ptr<Kernel> kernel, cl_uint work_dimension, size_t *g_work_offset, size_t *g_work_size, size_t *l_work_size, std::vector<std::shared_ptr<clw::Event>> wait_list);

            //Queues a buffer read
            ErrorOr<std::shared_ptr<Event>> read_buffer(Buffer& bfr, bool block, size_t buffer_offset, size_t read_size, void* target, std::vector<std::shared_ptr<clw::Event>> wait_list);

            //Queues a buffer write
            ErrorOr<std::shared_ptr<Event>> write_buffer(Buffer& bfr, bool block, size_t buffer_offset, size_t write_size, void* target, std::vector<std::shared_ptr<clw::Event>> wait_list);

            //Queues a image read
            ErrorOr<std::shared_ptr<Event>> read_image(
                Image& img, bool block, size_t * origin, size_t * region, const size_t row_pitch,
                const size_t slice_pitch, void* target, std::vector<std::shared_ptr<clw::Event>> wait_list
            );
            
            //Queues a image write
            ErrorOr<std::shared_ptr<Event>> write_image(
                Image& img, bool block, size_t * origin, size_t * region, const size_t row_pitch,
                const size_t slice_pitch, void* target, std::vector<std::shared_ptr<clw::Event>> wait_list
            );

        private:
            cl_command_queue queue_;
            std::shared_ptr<Context> context_;

            //Pulls out the cl_events from clw::event list
            static std::vector<cl_event> pull_out_events(std::vector<std::shared_ptr<clw::Event>> wait_list);
    };
}

#endif