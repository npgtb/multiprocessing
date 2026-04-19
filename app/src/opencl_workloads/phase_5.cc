#include <helpers.h>
#include <profiler.h>
#include <scope_timer.h>
#include <opencl_workloads/phase_5.h>

namespace mp_course::gpu_workloads::phase_5{

    //Loads the opencl file and initializes the given kernels from it
    cl_int initialize(OpenCLRuntime& runtime){
        ScopeTimer scope_timer("initialize");
        const std::vector<std::string> kernels = {
            "resize_image", "grayscale_image", "calculate_disparity_map", "cross_check_occulsion_disparity_maps"
        };
        const std::string program_path = "data/kernels/phase_5.cl";
        clw::Device device = prefered_device();
        return runtime.load_file(device, program_path, kernels);
    }

    //Create all the buffers for the pipeline
    cl_int create_buffers(OpenCLRuntime& runtime, std::vector<clw::Buffer>& buffers, const size_t image_size, const size_t image_byte_size, const size_t downscale_factor, const size_t downscaled_image_size){
        const size_t downscaled_byte_size = downscaled_image_size * sizeof(uint32_t);
        //Buffer descriptions
        std::vector<clw::BufferDescription> descriptions = {
            //Two Initial buffers for the original images
            {CL_MEM_READ_ONLY, image_byte_size},
            {CL_MEM_READ_ONLY, image_byte_size},

            //Two buffers for the downscaled images
            {CL_MEM_READ_WRITE, downscaled_byte_size},
            {CL_MEM_READ_WRITE, downscaled_byte_size},

            //Two buffers for the grayscaled images => uint8_t
            {CL_MEM_READ_WRITE, downscaled_image_size},
            {CL_MEM_READ_WRITE, downscaled_image_size},

            //Two buffers for the disparity maps
            {CL_MEM_READ_WRITE, downscaled_image_size},
            {CL_MEM_READ_WRITE, downscaled_image_size},

            //One buffer for the post processed result
            {CL_MEM_READ_WRITE, downscaled_image_size},
        };
        //Reserve buffer memory
        buffers.reserve(descriptions.size());
        //Create buffers
        return runtime.context->create_buffers(buffers, descriptions);
    }

    //Binds the constant arguments of the pipeline
    cl_int bind_pipeline_args(
        OpenCLRuntime& runtime, const int downscale_factor, const int original_width,
        const int window_radius, const int min_disparity, const int max_disparity,
        const int threshold_value
    ){
        //Kernels
        std::shared_ptr<clw::Kernel> downscale_kernel = runtime.kernels[0];
        std::shared_ptr<clw::Kernel> grayscale_kernel = runtime.kernels[1];
        std::shared_ptr<clw::Kernel> zncc_kernel = runtime.kernels[2];
        std::shared_ptr<clw::Kernel> postprocess_kernel = runtime.kernels[3];

        std::vector<std::pair<std::shared_ptr<clw::Kernel>, std::vector<std::pair<int, int>>>> kernel_binds = 
        {
            {
                    downscale_kernel, std::vector<std::pair<int,int>>{
                        {2, downscale_factor}, {3, original_width}
                    }
                },
            {
                    zncc_kernel, std::vector<std::pair<int,int>>{
                        {3, window_radius}, {4, min_disparity}, {5, max_disparity}
                    }
                },
            {
                    postprocess_kernel, std::vector<std::pair<int, int>>{
                        {3, threshold_value}, {4, window_radius}, {5, min_disparity}, {6, max_disparity}
                    }
                }
        };

        cl_int error_code = CL_SUCCESS;
        //Bind the arguments
        for(auto& kernel_arg_pair : kernel_binds){
            for(auto& index_arg_pair : kernel_arg_pair.second){
                if((error_code = kernel_arg_pair.first->bind_arg<int>(index_arg_pair.first, index_arg_pair.second, sizeof(int))) != CL_SUCCESS){
                    return error_code;
                }
            }
        }
        return CL_SUCCESS;
    }

    //Binds the given buffers to the kernel as arguments
    cl_int bind_buffers(std::shared_ptr<clw::Kernel> kernel, std::vector<std::pair<int, std::reference_wrapper<clw::Buffer>>> bind_data){
        cl_int error_code = CL_SUCCESS;
        //Bind the buffers to the kernel
        for(auto& bind_pair : bind_data){
            if((error_code = kernel->bind_arg(bind_pair.first, bind_pair.second)) != CL_SUCCESS){
                break;
            }
        }
        return error_code;
    }

    //Queue work into the command queueu based ont he given argumentation
    clw::ErrorOr<std::shared_ptr<clw::Event>> queue_work(
        OpenCLRuntime& runtime, std::shared_ptr<clw::Kernel> kernel, size_t work_dimensions, size_t * global_work_size, size_t * local_size,
        std::vector<std::shared_ptr<clw::Event>> conditions, std::vector<std::pair<int, std::reference_wrapper<clw::Buffer>>> bind_data
    ){
        cl_int error_code = CL_SUCCESS;
        if ((error_code = bind_buffers(kernel, bind_data)) != CL_SUCCESS) {
            return error_code;
        }
        //Start the kernel with buffer_write as wait condition
        return runtime.cc_queue->execute_kernel(kernel, work_dimensions, nullptr, global_work_size, local_size, conditions);
    }


    //Split pipeline rescale => grayscale, combine => Zncc
    cl_int pipeline(
        OpenCLRuntime& runtime, Image& left, Image& right, Image& map,
        const int downscale_factor, const int window_radius, const int min_disparity,
        const int max_disparity, const int threshold_value
    ){
        ScopeTimer scope_timer("pipeline");
        const int image_size = left.h * left.w;
        const size_t downscaled_width = left.w / downscale_factor;
        const size_t downscaled_height = left.h / downscale_factor;
        const size_t downscaled_image_size = downscaled_width * downscaled_height;
        const size_t image_byte_size = image_size * sizeof(uint32_t);

        Profiler::add_info(
            "Global work group (" + std::to_string(downscaled_width) + "," +
            std::to_string(downscaled_height) + ")"
        );
        
        //Create buffers for the pipeline and bind the pipeline
        cl_int error_code = CL_SUCCESS;
        std::vector<clw::Buffer> buffers;
        if(
            (error_code = create_buffers(runtime, buffers, image_size, image_byte_size, downscale_factor, downscaled_image_size)) != CL_SUCCESS ||
            (error_code = bind_pipeline_args(
                runtime, downscale_factor, left.w, 
                window_radius, min_disparity, max_disparity, threshold_value
            )) != CL_SUCCESS
        ){
            Profiler::add_info("Initial pipeline setup failed: " + std::to_string(error_code));
            return error_code;
        }

        //Kernels
        std::shared_ptr<clw::Kernel> downscale_kernel = runtime.kernels[0];
        std::shared_ptr<clw::Kernel> grayscale_kernel = runtime.kernels[1];
        std::shared_ptr<clw::Kernel> zncc_kernel = runtime.kernels[2];
        std::shared_ptr<clw::Kernel> postprocess_kernel = runtime.kernels[3];

        //First step of the pipeline queue the writes of the original images to the device memory
        clw::ErrorOr<std::shared_ptr<clw::Event>> left_write = runtime.cc_queue->write_buffer(buffers[0], false, 0, image_byte_size, left.pixels, {});
        clw::ErrorOr<std::shared_ptr<clw::Event>> right_write = runtime.cc_queue->write_buffer(buffers[1], false, 0, image_byte_size, right.pixels, {});
        //Queue action was successful?
        if(!left_write.ok()){
            Profiler::add_info("Left image write to buffer failed: " + std::to_string(left_write.error()));
            return left_write.error();
        }
        if(!right_write.ok()){
            Profiler::add_info("Right image write to buffer failed: " + std::to_string(right_write.error()));
            return right_write.error();
        }

        //Queue the downscale action on both images
        size_t  split_work_dimensions[2] = {downscaled_width, downscaled_height};

        clw::ErrorOr<std::shared_ptr<clw::Event>> left_resize = queue_work(runtime, downscale_kernel, 2, split_work_dimensions, nullptr, {left_write.value()}, {{0,buffers[0]}, {1,buffers[2]}});
        clw::ErrorOr<std::shared_ptr<clw::Event>> right_resize = queue_work(runtime, downscale_kernel, 2, split_work_dimensions, nullptr, {right_write.value()}, {{0,buffers[1]}, {1,buffers[3]}});
        //Queueing was succesful?
        if(!left_resize.ok()){
            Profiler::add_info("Left image downscale queueing failed: " + std::to_string(left_resize.error()));
            return left_resize.error();
        }
        if(!right_resize.ok()){
            Profiler::add_info("Right image downscale queueing failed: " + std::to_string(right_resize.error()));
            return right_resize.error();
        }

        //Queue the grayscale action on both images
        clw::ErrorOr<std::shared_ptr<clw::Event>> left_grayscale = queue_work(runtime, grayscale_kernel, 2, split_work_dimensions, nullptr, {left_resize.value()}, {{0,buffers[2]}, {1,buffers[4]}});
        clw::ErrorOr<std::shared_ptr<clw::Event>> right_grayscale = queue_work(runtime, grayscale_kernel, 2, split_work_dimensions, nullptr, {right_resize.value()}, {{0,buffers[3]}, {1,buffers[5]}});
        if(!left_grayscale.ok()){
            Profiler::add_info("Left image grayscaling queueing failed: " + std::to_string(left_grayscale.error()));
            return left_grayscale.error();
        }
        if(!right_grayscale.ok()){
            Profiler::add_info("Right image grayscaling queueing failed: " + std::to_string(right_grayscale.error()));
            return right_grayscale.error();
        }

        //Recombine the split pipelines together for the zncc
        clw::ErrorOr<std::shared_ptr<clw::Event>> zncc_left = queue_work(runtime, zncc_kernel, 2, split_work_dimensions, nullptr, {left_grayscale.value(), right_grayscale.value()}, {{0,buffers[4]}, {1,buffers[5]}, {2, buffers[6]}});
        clw::ErrorOr<std::shared_ptr<clw::Event>> zncc_right = queue_work(runtime, zncc_kernel, 2, split_work_dimensions, nullptr, {left_grayscale.value(), right_grayscale.value()}, {{0,buffers[5]}, {1,buffers[4]}, {2, buffers[7]}});
        if(!zncc_left.ok()){
            Profiler::add_info("Left Zncc queueing failed: " + std::to_string(zncc_left.error()));
            return zncc_left.error();
        }
        if(!zncc_right.ok()){
            Profiler::add_info("Right Zncc queueing failed: " + std::to_string(zncc_right.error()));
            return zncc_right.error();
        }

        //Queue the post process
        clw::ErrorOr<std::shared_ptr<clw::Event>> post_process = queue_work(runtime, postprocess_kernel, 2, split_work_dimensions, nullptr, {zncc_left.value(), zncc_right.value()}, {{0,buffers[6]}, {1,buffers[7]}, {2, buffers[8]}});
        if(!post_process.ok()){
            Profiler::add_info("Post process queuing failed: " + std::to_string(post_process.error()));
            return post_process.error();
        }

        //Allocate post process map
        map.free_memory();
        map.w = downscaled_width; map.h = downscaled_height;
        map.format = ImageFormat::GRAY;
        map.pixels = malloc(downscaled_image_size);

        //Read the map into ram
        clw::ErrorOr<std::shared_ptr<clw::Event>> buffer_read = runtime.cc_queue->read_buffer(buffers[8], false, 0, downscaled_image_size, map.pixels, {post_process.value()});
        if(!buffer_read.ok()){
            Profiler::add_info("Result reading from buffer failed: " + std::to_string(buffer_read.error()));
            return buffer_read.error();
        }

        //Wait for the gpu to finish
        clw::Event::wait({buffer_read.value()});

        //Pull out the timings from the pipeline events and report them
        report_cl_timing("Pipeline | write left", left_write.value());
        report_cl_timing("Pipeline | write right", right_write.value());
        report_cl_timing("Pipeline | downscale left", left_resize.value());
        report_cl_timing("Pipeline | downscale right", right_resize.value());
        report_cl_timing("Pipeline | grayscale left", left_grayscale.value());
        report_cl_timing("Pipeline | grayscale right", right_grayscale.value());
        report_cl_timing("Pipeline | Zncc map left", zncc_left.value());
        report_cl_timing("Pipeline | Zncc map right", zncc_right.value());
        report_cl_timing("Pipeline | Post process", post_process.value());
        report_cl_timing("Pipeline | Map read", buffer_read.value());
        
        return CL_SUCCESS;
    }

}
