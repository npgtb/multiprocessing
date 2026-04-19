#include <opencl_wrapper/command_queue.h>

namespace clw{
    //Initialize empty CommandQueue
    CommandQueue::CommandQueue():queue_(nullptr), context_(nullptr){}

    //Initialize CommandQueue with queue
    CommandQueue::CommandQueue(cl_command_queue queue, std::shared_ptr<Context> queue_context):queue_(queue), context_(queue_context){}

    //Copy constructors
    CommandQueue::CommandQueue(CommandQueue && other):queue_(other.queue_), context_(other.context_){
        other.queue_ = nullptr;
        other.context_ = nullptr;
    }

    //Release the memory
    CommandQueue::~CommandQueue(){
        release();
    }

    //Assignment operators
    CommandQueue& CommandQueue::operator=(CommandQueue&& other){
        queue_ = other.queue_;
        context_ = other.context_;
        other.queue_ = nullptr;
        other.context_ = nullptr;
        return *this;
    }

    //CCQueue kinda exists outside the destruction hierarchy, so manually release it
    void CommandQueue::release(){
        if(queue_){
            clReleaseCommandQueue(queue_);
            queue_ = nullptr;
            context_ = nullptr;
        }
    }

    //Queues the execution of the kernel
    ErrorOr<std::shared_ptr<Event>> CommandQueue::execute_kernel(
        std::shared_ptr<Kernel> kernel, cl_uint work_dimension, size_t *g_work_offset, size_t *g_work_size, size_t *l_work_size, std::vector<std::shared_ptr<clw::Event>> wait_list
    ){
        cl_int wait_events = 0;
        cl_event * c_wait_list = nullptr;
        std::vector<cl_event> wait_conditions = CommandQueue::pull_out_events(wait_list);
        //Pull out the wait events if there are any
        if(wait_conditions.size() > 0){
            wait_events = wait_conditions.size();
            c_wait_list = wait_conditions.data();
        }
        cl_event event = nullptr;
        //https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clEnqueueNDRangeKernel.html
        //Queue the execution of the kernel: ccqueue, kernel, work dim, g work offset, g work size, l work size, wait list size, wait list, query_obj
        cl_int error_code = clEnqueueNDRangeKernel(
            queue_, kernel->kernel_,  work_dimension, g_work_offset, g_work_size, l_work_size, wait_events, c_wait_list, &event 
        );
        if(error_code == CL_SUCCESS){
            return std::make_shared<Event>(event, shared_from_this());
        }
        return error_code;
    }

    //Queues a buffer read
    ErrorOr<std::shared_ptr<Event>> CommandQueue::read_buffer(Buffer& bfr, bool block, size_t buffer_offset, size_t read_size, void* target, std::vector<std::shared_ptr<clw::Event>> wait_list){
        cl_int wait_events = 0;
        cl_event * c_wait_list = nullptr;
        std::vector<cl_event> wait_conditions = CommandQueue::pull_out_events(wait_list);
        //Pull out the wait events if there are any
        if(wait_conditions.size() > 0){
            wait_events = wait_conditions.size();
            c_wait_list = wait_conditions.data();
        }
        
        cl_event event = nullptr;
        //https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clEnqueueReadBuffer.html
        //Read from the given buffer: queue, bfr, block read, device offset, read size, target, wait list size, wait list, query_obj
        cl_int error_code = clEnqueueReadBuffer(queue_, bfr.buffer_, block, buffer_offset, read_size, target, wait_events, c_wait_list, &event);
        if(error_code == CL_SUCCESS){
            return std::make_shared<Event>(event, shared_from_this());
        }
        return error_code;
    }

    //Queues a buffer write
    ErrorOr<std::shared_ptr<Event>> CommandQueue::write_buffer(Buffer& bfr, bool block, size_t buffer_offset, size_t write_size, void* target, std::vector<std::shared_ptr<clw::Event>> wait_list){
        cl_int wait_events = 0;
        cl_event * c_wait_list = nullptr;
        std::vector<cl_event> wait_conditions = CommandQueue::pull_out_events(wait_list);
        //Pull out the wait events if there are any
        if(wait_conditions.size() > 0){
            wait_events = wait_conditions.size();
            c_wait_list = wait_conditions.data();
        }
        
        cl_event event = nullptr;
        //https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clEnqueueReadBuffer.html
        //Read from the given buffer: queue, bfr, block write, device offset, write size, target, wait list size, wait list, query_obj
        cl_int error_code = clEnqueueWriteBuffer(queue_, bfr.buffer_, block, buffer_offset, write_size, target, wait_events, c_wait_list, &event);
        if(error_code == CL_SUCCESS){
            return std::make_shared<Event>(event, shared_from_this());
        }
        return error_code;
    }

    //Queues a image read
    ErrorOr<std::shared_ptr<Event>> CommandQueue::read_image(
        Image& img, bool block, size_t * origin, size_t * region, const size_t row_pitch,
        const size_t slice_pitch, void* target, std::vector<std::shared_ptr<clw::Event>> wait_list
    ){
        cl_int wait_events = 0;
        cl_event * c_wait_list = nullptr;
        std::vector<cl_event> wait_conditions = CommandQueue::pull_out_events(wait_list);
        //Pull out the wait events if there are any
        if(wait_conditions.size() > 0){
            wait_events = wait_conditions.size();
            c_wait_list = wait_conditions.data();
        }
        
        cl_event event = nullptr;
        //https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clEnqueueReadBuffer.html
        //Read from the given buffer: queue, bfr, block read, device offset, read size, target, wait list size, wait list, query_obj
        cl_int error_code = clEnqueueReadImage(queue_, img.image_, block, origin, region, row_pitch, slice_pitch, target, wait_events, c_wait_list, &event);
        if(error_code == CL_SUCCESS){
            return std::make_shared<Event>(event, shared_from_this());
        }
        return error_code;
    }

    //Queues a image write
    ErrorOr<std::shared_ptr<Event>> CommandQueue::write_image(
        Image& img, bool block, size_t * origin, size_t * region, const size_t row_pitch,
        const size_t slice_pitch, void* target, std::vector<std::shared_ptr<clw::Event>> wait_list
    ){
        cl_int wait_events = 0;
        cl_event * c_wait_list = nullptr;
        std::vector<cl_event> wait_conditions = CommandQueue::pull_out_events(wait_list);
        //Pull out the wait events if there are any
        if(wait_conditions.size() > 0){
            wait_events = wait_conditions.size();
            c_wait_list = wait_conditions.data();
        }
        
        cl_event event = nullptr;
        //https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clEnqueueWriteImage.html
        //cc_queu, image, blocking, origin, region, row_pitch, slice_pitch, data, wait_count, wait_list, event
        cl_int error_code = clEnqueueWriteImage(queue_, img.image_, block, origin, region, row_pitch, slice_pitch, target, wait_events, c_wait_list, &event);
        if(error_code == CL_SUCCESS){
            return std::make_shared<Event>(event, shared_from_this());
        }
        return error_code;
    }


    //Pulls out the cl_events from clw::event list
    std::vector<cl_event> CommandQueue::pull_out_events(std::vector<std::shared_ptr<clw::Event>> wait_list){
        std::vector<cl_event> wait_conditions;
        //Pull out the wait events if there are any
        if(wait_list.size() > 0){
            wait_conditions.reserve(wait_list.size());
            for(auto& event : wait_list){
                wait_conditions.push_back(event->event_);
            }
        }
        return wait_conditions;
    }

}