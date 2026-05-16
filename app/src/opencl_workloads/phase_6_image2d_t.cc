#include <helpers.h>
#include <profiler.h>
#include <scope_timer.h>
#include <opencl_workloads/phase_6_image2d_t.h>

namespace mp::gpu_workloads::phase_6_image2d_t{

    //Loads the opencl file and initializes the given kernels from it
    cl_int initialize(OpenCLRuntime& runtime){
        ScopeTimer scope_timer("initialize");
        const std::vector<std::string> kernels = {
            "resize_image", "grayscale_image", "calculate_disparity_map", "cross_check_occulsion_disparity_maps"
        };
        const std::string program_path = "data/kernels/phase_6_image_2d_t.cl";
        clw::Device device = prefered_device();
        std::vector<clw::Program::CompileOption> compile_options{};
        return runtime.load_file(device, program_path, kernels, compile_options);
    }

    //Create all the buffers for the pipeline
    cl_int create_images(
        OpenCLRuntime& runtime, std::vector<clw::Image>& images, const size_t image_width, const size_t image_height,
        const size_t downscaled_image_width, const size_t downscaled_image_height
    ){

        std::vector<clw::ImageDescription> descriptions = {
            //Two Initial images for the original images
            {
                .memory_flags = CL_MEM_READ_ONLY,
                .format = {.image_channel_order = CL_RGBA, .image_channel_data_type = CL_UNSIGNED_INT8},
                .description = {.image_type = CL_MEM_OBJECT_IMAGE2D, .image_width = image_width, .image_height = image_height, .image_row_pitch = 0},
                .host_data = nullptr,
            },
            {
                .memory_flags = CL_MEM_READ_ONLY,
                .format = {.image_channel_order = CL_RGBA, .image_channel_data_type = CL_UNSIGNED_INT8},
                .description = {.image_type = CL_MEM_OBJECT_IMAGE2D, .image_width = image_width, .image_height = image_height, .image_row_pitch = 0},
                .host_data = nullptr,
            },
            //Two images for the downscaled images
            {
                .memory_flags = CL_MEM_READ_WRITE,
                .format = {.image_channel_order = CL_RGBA, .image_channel_data_type = CL_UNSIGNED_INT8},
                .description = {.image_type = CL_MEM_OBJECT_IMAGE2D, .image_width = downscaled_image_width, .image_height = downscaled_image_height, .image_row_pitch = 0},
                .host_data = nullptr,
            },
            {
                .memory_flags = CL_MEM_READ_WRITE,
                .format = {.image_channel_order = CL_RGBA, .image_channel_data_type = CL_UNSIGNED_INT8},
                .description = {.image_type = CL_MEM_OBJECT_IMAGE2D, .image_width = downscaled_image_width, .image_height = downscaled_image_height, .image_row_pitch = 0},
                .host_data = nullptr,
            },
            //Two images for the grayscaled images => uint8_t
            {
                .memory_flags = CL_MEM_READ_WRITE,
                .format = {.image_channel_order = CL_R, .image_channel_data_type = CL_UNSIGNED_INT8},
                .description = {.image_type = CL_MEM_OBJECT_IMAGE2D, .image_width = downscaled_image_width, .image_height = downscaled_image_height, .image_row_pitch = 0},
                .host_data = nullptr,
            },
            {
                .memory_flags = CL_MEM_READ_WRITE,
                .format = {.image_channel_order = CL_R, .image_channel_data_type = CL_UNSIGNED_INT8},
                .description = {.image_type = CL_MEM_OBJECT_IMAGE2D, .image_width = downscaled_image_width, .image_height = downscaled_image_height, .image_row_pitch = 0},
                .host_data = nullptr,
            },
            //Two images for the disparity maps
            {
                .memory_flags = CL_MEM_READ_WRITE,
                .format = {.image_channel_order = CL_R, .image_channel_data_type = CL_UNSIGNED_INT8},
                .description = {.image_type = CL_MEM_OBJECT_IMAGE2D, .image_width = downscaled_image_width, .image_height = downscaled_image_height, .image_row_pitch = 0},
                .host_data = nullptr,
            },
            {
                .memory_flags = CL_MEM_READ_WRITE,
                .format = {.image_channel_order = CL_R, .image_channel_data_type = CL_UNSIGNED_INT8},
                .description = {.image_type = CL_MEM_OBJECT_IMAGE2D, .image_width = downscaled_image_width, .image_height = downscaled_image_height, .image_row_pitch = 0},
                .host_data = nullptr,
            },
            //One image for the post processed result
            {
                .memory_flags = CL_MEM_READ_WRITE,
                .format = {.image_channel_order = CL_R, .image_channel_data_type = CL_UNSIGNED_INT8},
                .description = {.image_type = CL_MEM_OBJECT_IMAGE2D, .image_width = downscaled_image_width, .image_height = downscaled_image_height, .image_row_pitch = 0},
                .host_data = nullptr,
            }, 
        };
        //Reserve buffer memory
        images.reserve(descriptions.size());
        //Create buffers
        return runtime.context->create_images(images, descriptions);
    }

    //Bind arguments to the kernel
    cl_int bind_args(std::shared_ptr<clw::Kernel> kernel, std::vector<std::pair<int, int>> args){
        cl_int error_code = CL_SUCCESS;
        //Bind the arguments
        for(auto& index_arg_pair : args){
            if((error_code = kernel->bind_arg<int>(index_arg_pair.first, index_arg_pair.second, sizeof(int))) != CL_SUCCESS){
                return error_code;
            }
        }
        return CL_SUCCESS;
    }

    //Binds the constant arguments of the pipeline
    cl_int bind_pipeline_args(
        OpenCLRuntime& runtime, const int downscale_factor, const int window_radius, 
        const int min_disparity, const int max_disparity, const int threshold_value
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
                        {2, downscale_factor}
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
            if((error_code = bind_args(kernel_arg_pair.first, kernel_arg_pair.second)) != CL_SUCCESS){
                return error_code;
            }
        }
        return CL_SUCCESS;
    }

    //Binds the given buffers to the kernel as arguments
    cl_int bind_images(std::shared_ptr<clw::Kernel> kernel, std::vector<std::pair<int, std::reference_wrapper<clw::Image>>> bind_data){
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
        std::vector<std::shared_ptr<clw::Event>> conditions, std::vector<std::pair<int, std::reference_wrapper<clw::Image>>> bind_data,
        std::vector<std::pair<int, int>> args
    ){
        cl_int error_code = CL_SUCCESS;
        //Bind arguments
        if ((error_code = bind_args(kernel, args)) != CL_SUCCESS) {
            return error_code;
        }
        //Bind images
        if ((error_code = bind_images(kernel, bind_data)) != CL_SUCCESS) {
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
        const size_t scaled_w = left.w / downscale_factor;
        const size_t scaled_h = left.h / downscale_factor;
        const size_t scaled_bsize = scaled_w * scaled_h;
        size_t origin[3] = {0, 0, 0}; 
        size_t original_region[3] = {static_cast<size_t>(left.w), static_cast<size_t>(left.h), 1};
        size_t scaled_region[3] = {scaled_w, scaled_h, 1};

        //Create images for the pipeline and bind the pipeline
        cl_int error_code = CL_SUCCESS;
        std::vector<clw::Image> images;
        if(
            (error_code = create_images(runtime, images, left.w, left.h, scaled_w, scaled_h)) != CL_SUCCESS ||
            (error_code = bind_pipeline_args(
                runtime, downscale_factor, window_radius, 
                min_disparity, max_disparity, threshold_value
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

        //Images named
        clw::Image& left_orig = images[0];
        clw::Image& right_orig = images[1];

        clw::Image& left_down = images[2];
        clw::Image& right_down = images[3];

        clw::Image& left_gray = images[4];
        clw::Image& right_gray = images[5];

        clw::Image& left_dis = images[6];
        clw::Image& right_dis = images[7];
        clw::Image& post_pros = images[8];

        //First step of the pipeline queue the writes of the original images to the device memory
        clw::ErrorOr<std::shared_ptr<clw::Event>> left_write = runtime.cc_queue->write_image(left_orig, false, origin, original_region, 0, 0, left.data<uint32_t>(), {});
        clw::ErrorOr<std::shared_ptr<clw::Event>> right_write = runtime.cc_queue->write_image(right_orig, false, origin, original_region, 0, 0, right.data<uint32_t>(), {});
        //Queue action was successful?
        if(!left_write.ok()){
            Profiler::add_info("Left image write to image failed: " + std::to_string(left_write.error()));
            return left_write.error();
        }
        if(!right_write.ok()){
            Profiler::add_info("Right image write to image failed: " + std::to_string(right_write.error()));
            return right_write.error();
        }

        //Queue the downscale action on both images
        size_t  split_work_dimensions[2] = {scaled_w, scaled_h};

        clw::ErrorOr<std::shared_ptr<clw::Event>> left_resize = queue_work(runtime, downscale_kernel, 2, split_work_dimensions, nullptr, {left_write.value()}, {{0,left_orig}, {1,left_down}}, {});
        clw::ErrorOr<std::shared_ptr<clw::Event>> right_resize = queue_work(runtime, downscale_kernel, 2, split_work_dimensions, nullptr, {right_write.value()}, {{0,right_orig}, {1,right_down}}, {});
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
        clw::ErrorOr<std::shared_ptr<clw::Event>> left_grayscale = queue_work(runtime, grayscale_kernel, 2, split_work_dimensions, nullptr, {left_resize.value()}, {{0,left_down}, {1,left_gray}}, {});
        clw::ErrorOr<std::shared_ptr<clw::Event>> right_grayscale = queue_work(runtime, grayscale_kernel, 2, split_work_dimensions, nullptr, {right_resize.value()}, {{0,right_down}, {1,right_gray}}, {});
        if(!left_grayscale.ok()){
            Profiler::add_info("Left image grayscaling queueing failed: " + std::to_string(left_grayscale.error()));
            return left_grayscale.error();
        }
        if(!right_grayscale.ok()){
            Profiler::add_info("Right image grayscaling queueing failed: " + std::to_string(right_grayscale.error()));
            return right_grayscale.error();
        }

        //Recombine the split pipelines together for the zncc
        clw::ErrorOr<std::shared_ptr<clw::Event>> zncc_left = queue_work(runtime, zncc_kernel, 2, split_work_dimensions, nullptr, {left_grayscale.value(), right_grayscale.value()}, {{0,left_gray}, {1,right_gray}, {2, left_dis}}, {{6, -1}});
        clw::ErrorOr<std::shared_ptr<clw::Event>> zncc_right = queue_work(runtime, zncc_kernel, 2, split_work_dimensions, nullptr, {left_grayscale.value(), right_grayscale.value()}, {{0,right_gray}, {1,left_gray}, {2, right_dis}}, {{6, 1}});
        if(!zncc_left.ok()){
            Profiler::add_info("Left Zncc queueing failed: " + std::to_string(zncc_left.error()));
            return zncc_left.error();
        }
        if(!zncc_right.ok()){
            Profiler::add_info("Right Zncc queueing failed: " + std::to_string(zncc_right.error()));
            return zncc_right.error();
        }

        //Queue the post process
        clw::ErrorOr<std::shared_ptr<clw::Event>> pp_queu = queue_work(runtime, postprocess_kernel, 2, split_work_dimensions, nullptr, {zncc_left.value(), zncc_right.value()}, {{0,left_dis}, {1,right_dis}, {2, post_pros}}, {});
        if(!pp_queu.ok()){
            Profiler::add_info("Post process queuing failed: " + std::to_string(pp_queu.error()));
            return pp_queu.error();
        }

        //Allocate post process map
        map.init<uint8_t>(scaled_w, scaled_h, ImageFormat::GRAY);
        //Read the map into ram
        clw::ErrorOr<std::shared_ptr<clw::Event>> image_read = runtime.cc_queue->read_image(post_pros, false, origin, scaled_region, 0, 0, map.data<uint8_t>(), {pp_queu.value()});
        if(!image_read.ok()){
            Profiler::add_info("Result reading from image failed: " + std::to_string(image_read.error()));
            return image_read.error();
        }

        //Wait for the gpu to finish
        clw::Event::wait({image_read.value()});

        //Pull out the timings from the pipeline events and report them
        report_cl_timing("Pipeline | write left", left_write.value());
        report_cl_timing("Pipeline | write right", right_write.value());
        report_cl_timing("Pipeline | downscale left", left_resize.value());
        report_cl_timing("Pipeline | downscale right", right_resize.value());
        report_cl_timing("Pipeline | grayscale left", left_grayscale.value());
        report_cl_timing("Pipeline | grayscale right", right_grayscale.value());
        report_cl_timing("Pipeline | Zncc map left", zncc_left.value());
        report_cl_timing("Pipeline | Zncc map right", zncc_right.value());
        report_cl_timing("Pipeline | Post process", pp_queu.value());
        report_cl_timing("Pipeline | Map read", image_read.value());
        
        return CL_SUCCESS;
    }

}
