#include <helpers.h>
#include <profiler.h>
#include <scope_timer.h>
#include <opencl_workloads/phase_7_integral_aps_cl.h>

namespace mp::gpu_workloads::phase_7_integral_aps_cl{

    //Loads the opencl file and initializes the given kernels from it
    cl_int initialize(
        OpenCLRuntime& runtime, const int window_radius, const int min_disparity,
        const int max_disparity, const int threshold_value
    ){
        ScopeTimer scope_timer("initialize");
        const std::vector<std::string> kernels = {
            "resize_image", "grayscale_image", "calculate_integral_map_rows",
             "calculate_integral_map_columns", "calculate_disparity_map", "cross_check_occulsion_disparity_maps"
        };
        const std::string program_path = "data/kernels/phase_7_integral_aps_cl.cl";
        clw::Device device = prefered_device();
        std::vector<clw::Program::CompileOption> compile_options{
            {.name="DRADIUS", .value=std::to_string(window_radius)},
            {.name="DMAX_D_EXC", .value=std::to_string(max_disparity)},
            {.name="DMAX_D_INC", .value=std::to_string(max_disparity + 1)},
            {.name="DMIN_D", .value=std::to_string(min_disparity)},
            {.name="DTHRESHOLD_VALUE", .value=std::to_string(threshold_value)},
            {.name="cl-fast-relaxed-math"},
            {.name="cl-mad-enable"},
            {.name="cl-no-signed-zeros"},
            {.name="cl-unsafe-math-optimizations"},
            {.name="Werror"},
            {.name="cl-std", .value="CL2.0"}
        };
        return runtime.load_file(device, program_path, kernels, compile_options);
    }

    //Create all the buffers for the pipeline
    cl_int create_buffers(
        OpenCLRuntime& runtime, std::vector<clw::Buffer>& buffers, const size_t image_width, const size_t image_height,
        const size_t downscale_factor, const size_t group_width, const size_t group_height, const size_t sm_data_size
    ){
        const size_t scaled_w = image_width / downscale_factor;
        const size_t scaled_h = image_height / downscale_factor;
        const size_t scaled_w_padded = scaled_w + ((group_width - (scaled_w % group_width)) % group_width);
        const size_t scaled_h_padded = scaled_h + ((group_height - (scaled_h % group_height)) % group_height);
        const size_t original_size = image_width * image_height;
        const size_t original_bsize = original_size * sizeof(uint32_t);
      
        const size_t scaled_bsize = scaled_w * scaled_h;
        const size_t scaled_bsize_padded = scaled_w_padded * scaled_h_padded;
        const size_t downscaled_byte_size = scaled_bsize * sizeof(uint32_t);

        //Buffer descriptions
        std::vector<clw::BufferDescription> descriptions = {
            //Two Initial buffers for the original images
            {CL_MEM_READ_ONLY, original_bsize},
            {CL_MEM_READ_ONLY, original_bsize},

            //Two buffers for the downscaled images
            {CL_MEM_READ_WRITE, downscaled_byte_size},
            {CL_MEM_READ_WRITE, downscaled_byte_size},

            //Two buffers for the grayscaled images => uint8_t
            {CL_MEM_READ_WRITE, scaled_bsize_padded},
            {CL_MEM_READ_WRITE, scaled_bsize_padded},

            //Two buffers for the integral sum maps
            {CL_MEM_READ_WRITE, sm_data_size},
            {CL_MEM_READ_WRITE, sm_data_size},
            //Two buffers for the integral square maps
            {CL_MEM_READ_WRITE, sm_data_size},
            {CL_MEM_READ_WRITE, sm_data_size},

            //Two buffers for the disparity maps
            {CL_MEM_READ_WRITE, scaled_bsize_padded},
            {CL_MEM_READ_WRITE, scaled_bsize_padded},

            //One buffer for the post processed result
            {CL_MEM_READ_WRITE, scaled_bsize_padded},
        };
        //Reserve buffer memory
        buffers.reserve(descriptions.size());
        //Create buffers
        return runtime.context->create_buffers(buffers, descriptions);
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
        OpenCLRuntime& runtime, const int downscale_factor, const int original_width,
        const int window_radius, const int min_disparity, const int max_disparity,
        const int threshold_value, const int scaled_w, const int scaled_h,
        const int integ_width, const int integ_height
    ){
        //Kernels
        std::shared_ptr<clw::Kernel> downscale_kernel = runtime.kernels[0];
        std::shared_ptr<clw::Kernel> grayscale_kernel = runtime.kernels[1];
        std::shared_ptr<clw::Kernel> integral_row = runtime.kernels[2];
        std::shared_ptr<clw::Kernel> integral_column = runtime.kernels[3];
        std::shared_ptr<clw::Kernel> zncc_kernel = runtime.kernels[4];
        std::shared_ptr<clw::Kernel> postprocess_kernel = runtime.kernels[5];

        std::vector<std::pair<std::shared_ptr<clw::Kernel>, std::vector<std::pair<int, int>>>> kernel_binds = 
        {
        {
                downscale_kernel, std::vector<std::pair<int,int>>{
                    {2, downscale_factor}, {3, original_width}
                }
            },
            {
                integral_row, std::vector<std::pair<int,int>>{
                    {3, scaled_w}, {4, scaled_h}, {5, integ_width},
                }
            },
            {
                integral_column, std::vector<std::pair<int, int>>{
                    {2, integ_height}, {3, integ_width}
                }
            },
        {
                zncc_kernel, std::vector<std::pair<int,int>>{
                    {8, scaled_w}, {9, scaled_h},
                    {10, integ_width}, {11, integ_height}
                }
            },
        {
                postprocess_kernel, std::vector<std::pair<int, int>>{
                     {3, scaled_w}, {4, scaled_h}
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

    //Allocates the memory to the __local buffers in kernels
    cl_int allocate_local_memory(
        OpenCLRuntime& runtime, const int window_radius, const int min_disparity, 
        const int max_disparity, const int dis_group_width, const int dis_group_height
    ){
        const size_t disparity_range = (max_disparity - min_disparity);
        const size_t dis_tile_height = (window_radius * 2 + dis_group_height);
        const size_t dis_tile_left_width = (window_radius * 2 + dis_group_width);
        const size_t dis_tile_right_width = (window_radius * 2 + dis_group_width + disparity_range);
        const size_t dis_tile_left_size = dis_tile_left_width * dis_tile_height;
        const size_t dis_tile_right_size = dis_tile_right_width * dis_tile_height;

        Profiler::add_info("Window radious / Disparity range: (" + std::to_string(window_radius) + "," + std::to_string(disparity_range) + ")");
        Profiler::add_info(
            "Local memory tile size L/R (" + std::to_string(dis_tile_left_width) + "," + std::to_string(dis_tile_height) + "), " +
            "(" + std::to_string(dis_tile_right_width) + "," + std::to_string(dis_tile_height) + ")"
        );
        Profiler::add_info("Local memory allocation L/R (" + std::to_string(dis_tile_left_size) + "," + std::to_string(dis_tile_right_size) + ")");

        std::shared_ptr<clw::Kernel> integ_row_kernel = runtime.kernels[2];
        std::shared_ptr<clw::Kernel> integ_column_kernel = runtime.kernels[3];
        std::shared_ptr<clw::Kernel> zncc_kernel = runtime.kernels[4];
        std::shared_ptr<clw::Kernel> postprocess_kernel = runtime.kernels[5];
        std::vector<std::pair<std::shared_ptr<clw::Kernel>, std::vector<std::pair<int, int>>>> kernel_binds = 
        {
        {
                zncc_kernel, std::vector<std::pair<int,int>>{
                    {14, dis_tile_left_size}, {15, dis_tile_right_size}
                }
            },
        {
                postprocess_kernel, std::vector<std::pair<int, int>>{
                    {5, dis_tile_left_size}, {6, dis_tile_right_size}
                }
            }
        };

        cl_int error_code = CL_SUCCESS;
        //Bind the arguments
        for(auto& kernel_arg_pair : kernel_binds){
            for(auto& index_arg_pair : kernel_arg_pair.second){
                clw::Local local(index_arg_pair.second);
                if((error_code = kernel_arg_pair.first->bind_arg(index_arg_pair.first, local)) != CL_SUCCESS){
                    Profiler::add_info("Failed to initiate local buffers: " + std::to_string(error_code));
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
        std::vector<std::shared_ptr<clw::Event>> conditions, std::vector<std::pair<int, std::reference_wrapper<clw::Buffer>>> bind_data,
        std::vector<std::pair<int, int>> args
    ){
        cl_int error_code = CL_SUCCESS;
        //Bind arguments
        if ((error_code = bind_args(kernel, args)) != CL_SUCCESS) {
            return error_code;
        }
        //Bind buffers
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
        const size_t group_width = 16;
        const size_t group_height = 8;

        const size_t original_size = left.h * left.w;
        const size_t scaled_w = left.w / downscale_factor;
        const size_t scaled_h = left.h / downscale_factor;
        const size_t scaled_w_padded = scaled_w + ((group_width - (scaled_w % group_width)) % group_width);
        const size_t scaled_h_padded = scaled_h + ((group_height - (scaled_h % group_height)) % group_height);
        const size_t scaled_bsize = scaled_w * scaled_h;
        const size_t original_bsize = original_size * sizeof(uint32_t);

        const size_t window_diameter = (window_radius << 1);
        const size_t integ_width = (scaled_w + window_diameter + max_disparity);
        const size_t integ_height = (scaled_h + window_diameter);
        const size_t integ_left_pad = window_radius;
        const size_t integ_right_pad = window_radius + max_disparity;
        const size_t integ_buffer_size = integ_width * integ_height * sizeof(uint64_t);
        const size_t integ_scan_width = 128;

        size_t lwork_dim_integral[1] = {integ_scan_width};
        size_t gwork_dim_integral_row[1] = {integ_scan_width * integ_height};
        size_t gwork_dim_integral_column[1] = {integ_scan_width * integ_width};
        size_t gwork_dim_scaled[2] = {scaled_w, scaled_h};
        size_t gwork_dim_scaled_pad[2] = {scaled_w_padded, scaled_h_padded};
        size_t local_work_group[2] = {group_width, group_height};

        //Create buffers for the pipeline and bind the pipeline
        cl_int error_code = CL_SUCCESS;
        std::vector<clw::Buffer> buffers;
        if(
            (
                error_code = create_buffers(
                    runtime, buffers, left.w, left.h, downscale_factor, 
                    group_width, group_height, integ_buffer_size
                )
            ) != CL_SUCCESS ||
            (
                error_code = bind_pipeline_args(
                    runtime, downscale_factor, left.w, 
                    window_radius, min_disparity, max_disparity, threshold_value,
                    scaled_w, scaled_h, integ_width, integ_height
                )
            ) != CL_SUCCESS ||
            (error_code = allocate_local_memory(runtime, window_radius, min_disparity, max_disparity, group_width, group_height)) != CL_SUCCESS
        ){
            Profiler::add_info("Initial pipeline setup failed: " + std::to_string(error_code));
            return error_code;
        }

        //Kernels
        std::shared_ptr<clw::Kernel> downscale_kernel = runtime.kernels[0];
        std::shared_ptr<clw::Kernel> grayscale_kernel = runtime.kernels[1];
        std::shared_ptr<clw::Kernel> integral_row = runtime.kernels[2];
        std::shared_ptr<clw::Kernel> integral_column = runtime.kernels[3];
        std::shared_ptr<clw::Kernel> zncc_kernel = runtime.kernels[4];
        std::shared_ptr<clw::Kernel> postprocess_kernel = runtime.kernels[5];

        //Buffers named
        clw::Buffer& left_orig = buffers[0];
        clw::Buffer& right_orig = buffers[1];

        clw::Buffer& left_down = buffers[2];
        clw::Buffer& right_down = buffers[3];

        clw::Buffer& left_gray = buffers[4];
        clw::Buffer& right_gray = buffers[5];

        clw::Buffer& left_ism = buffers[6];
        clw::Buffer& right_ism = buffers[7];

        clw::Buffer& left_isq = buffers[8];
        clw::Buffer& right_isq = buffers[9];

        clw::Buffer& left_dis = buffers[10];
        clw::Buffer& right_dis = buffers[11];

        clw::Buffer& post_pros = buffers[12];

        //First step of the pipeline queue the writes of the original images to the device memory
        clw::ErrorOr<std::shared_ptr<clw::Event>> left_write = runtime.cc_queue->write_buffer(left_orig, false, 0, original_bsize, left.data<uint32_t>(), {});
        clw::ErrorOr<std::shared_ptr<clw::Event>> right_write = runtime.cc_queue->write_buffer(right_orig, false, 0, original_bsize, right.data<uint32_t>(), {});
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
        clw::ErrorOr<std::shared_ptr<clw::Event>> left_resize = queue_work(runtime, downscale_kernel, 2, gwork_dim_scaled, nullptr, {left_write.value()}, {{0, left_orig}, {1, left_down}}, {});
        clw::ErrorOr<std::shared_ptr<clw::Event>> right_resize = queue_work(runtime, downscale_kernel, 2, gwork_dim_scaled, nullptr, {right_write.value()}, {{0, right_orig}, {1, right_down}}, {});
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
        clw::ErrorOr<std::shared_ptr<clw::Event>> left_grayscale = queue_work(runtime, grayscale_kernel, 2, gwork_dim_scaled, nullptr, {left_resize.value()}, {{0, left_down}, {1, left_gray}}, {});
        clw::ErrorOr<std::shared_ptr<clw::Event>> right_grayscale = queue_work(runtime, grayscale_kernel, 2, gwork_dim_scaled, nullptr, {right_resize.value()}, {{0, right_down}, {1, right_gray}}, {});
        if(!left_grayscale.ok()){
            Profiler::add_info("Left image grayscaling queueing failed: " + std::to_string(left_grayscale.error()));
            return left_grayscale.error();
        }
        if(!right_grayscale.ok()){
            Profiler::add_info("Right image grayscaling queueing failed: " + std::to_string(right_grayscale.error()));
            return right_grayscale.error();
        }

        //Queue the integral map row calc
        clw::ErrorOr<std::shared_ptr<clw::Event>> left_integral_row = queue_work(runtime, integral_row, 1, gwork_dim_integral_row, lwork_dim_integral, {left_grayscale.value()}, {{0, left_gray}, {1, left_ism}, {2, left_isq}}, {{6, 0}});
        clw::ErrorOr<std::shared_ptr<clw::Event>> right_integral_row = queue_work(runtime, integral_row, 1, gwork_dim_integral_row, lwork_dim_integral, {right_grayscale.value()}, {{0, right_gray}, {1, right_ism}, {2, right_isq}}, {{6, 1}});

        //Queueing was succesful?
        if(!left_integral_row.ok()){
            Profiler::add_info("Left integral row queueu failed: " + std::to_string(left_integral_row.error()));
            return left_integral_row.error();
        }
        if(!right_integral_row.ok()){
            Profiler::add_info("Right integral row queueu failed: " + std::to_string(right_integral_row.error()));
            return right_integral_row.error();
        }

        //Queue the integral map column calc
        clw::ErrorOr<std::shared_ptr<clw::Event>> left_integral_column = queue_work(runtime, integral_column, 1, gwork_dim_integral_column, lwork_dim_integral, {left_integral_row.value()}, {{0, left_ism}, {1, left_isq}}, {});
        clw::ErrorOr<std::shared_ptr<clw::Event>> right_integral_column = queue_work(runtime, integral_column, 1, gwork_dim_integral_column, lwork_dim_integral, {right_integral_row.value()}, {{0, right_ism}, {1, right_isq}}, {});

        //Queueing was succesful?
        if(!left_integral_column.ok()){
            Profiler::add_info("Left integral column queueu failed: " + std::to_string(left_integral_column.error()));
            return left_integral_column.error();
        }
        if(!right_integral_column.ok()){
            Profiler::add_info("Right integral column queueu failed: " + std::to_string(right_integral_column.error()));
            return right_integral_column.error();
        }

       //Recombine the split pipelines together for the zncc
        clw::ErrorOr<std::shared_ptr<clw::Event>> zncc_left = queue_work(runtime, zncc_kernel, 2, gwork_dim_scaled_pad, local_work_group, {left_integral_column.value(), right_integral_column.value()}, {{0,left_gray}, {1,right_gray}, {2, left_ism}, {3, right_ism}, {4, left_isq}, {5, right_isq}, {6, left_dis}},{{7, -1}, {12, integ_left_pad}, {13, integ_right_pad}});
        clw::ErrorOr<std::shared_ptr<clw::Event>> zncc_right = queue_work(runtime, zncc_kernel, 2, gwork_dim_scaled_pad, local_work_group, {left_integral_column.value(), right_integral_column.value()}, {{0,right_gray}, {1,left_gray}, {2, right_ism}, {3, left_ism}, {4, right_isq}, {5, left_isq}, {6, right_dis}}, {{7, 1}, {12, integ_right_pad}, {13, integ_left_pad}});
        if(!zncc_left.ok()){
            Profiler::add_info("Left Zncc queueing failed: " + std::to_string(zncc_left.error()));
            return zncc_left.error();
        }
        if(!zncc_right.ok()){
            Profiler::add_info("Right Zncc queueing failed: " + std::to_string(zncc_right.error()));
            return zncc_right.error();
        }

        //Queue the post process
        clw::ErrorOr<std::shared_ptr<clw::Event>> pp_queue = queue_work(runtime, postprocess_kernel, 2, gwork_dim_scaled_pad, local_work_group, {zncc_left.value(), zncc_right.value()}, {{0,left_dis}, {1,right_dis}, {2, post_pros}}, {});
        if(!pp_queue.ok()){
            Profiler::add_info("Post process queuing failed: " + std::to_string(pp_queue.error()));
            return pp_queue.error();
        }

        //Allocate post process map
        map.init<uint8_t>(scaled_w, scaled_h, ImageFormat::GRAY);
        //Read the map into ram
        clw::ErrorOr<std::shared_ptr<clw::Event>> buffer_read = runtime.cc_queue->read_buffer(post_pros, false, 0, scaled_bsize, map.data<uint8_t>(), {pp_queue.value()});
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
        report_cl_timing("Pipeline | Integral left row", left_integral_row.value());
        report_cl_timing("Pipeline | Integral right row", right_integral_row.value());
        report_cl_timing("Pipeline | Integral left column", left_integral_column.value());
        report_cl_timing("Pipeline | Integral right column", right_integral_column.value());
        report_cl_timing("Pipeline | Zncc map left", zncc_left.value());
        report_cl_timing("Pipeline | Zncc map right", zncc_right.value());
        report_cl_timing("Pipeline | Post process", pp_queue.value());
        report_cl_timing("Pipeline | Map read", buffer_read.value());
        
        return CL_SUCCESS;
    }

}
