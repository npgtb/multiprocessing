#include <profiler.h>
#include <opencl_workloads/phase_1.h>
#include <opencl_workloads/phase_2.h>
#include <opencl_workloads/phase_5.h>
#include <opencl_workloads/phase_6_tiled.h>
#include <opencl_workloads/phase_6_vectorized.h>
#include <opencl_workloads/phase_6_image2d_t.h>
#include <opencl_workloads/phase_7.h>
#include <opencl_workloads/opencl_workloads.h>

namespace mp::gpu_workloads{
    //Runs the opencl hello world program
    void run_hello_world_workload(){
        namespace phase_1 = mp::gpu_workloads::phase_1;
        mp::Profiler::segment_start("PHASE 1 - opencl_workload_hello_world");
        phase_1::hello_world();
    }

    //Runs the list info workload
    void run_list_info_workload(){
        namespace phase_2 = mp::gpu_workloads::phase_2;
        mp::Profiler::segment_start("PHASE 2 - opencl_workload_list_info");
        phase_2::list_info();
    }

    //Runs the matrix addition workload
    void run_matrix_addtion_workload(float * m1, float * m2, const int matrix_size, const int sample_count){
        cl_int error_code = 0;
        OpenCLRuntime runtime;
        namespace phase_2 = mp::gpu_workloads::phase_2;
        mp::Profiler::segment_start("PHASE 2 - opencl_workload_add_matrix");
        if((error_code = phase_2::initialize(runtime)) == CL_SUCCESS){
            for(int i = 0; i < sample_count; ++i){
                phase_2::add_matrix_pipeline(runtime, m1, m2, matrix_size);
            }
        }
        else{
            Profiler::add_info("Matrix addition | Failed to initialize opencl runtime: " + std::to_string(error_code));
        }
    }

    //Runs the matrix addition workload
    void run_zncc_pipeline_workload(
        const std::string &stereo_left, const std::string &stereo_right, const int downscale_factor, const int window_radius, 
        const int min_disparity, const int max_disparity, const int threshold_value, const int sample_count
    ){
        mp::Profiler::segment_start("PHASE 5 - ZNCC_OPENCL_GLOBAL");
        mp::Image left, right, pp_dmap;
        //Load stereo images
        if(left.load_path(stereo_left) && right.load_path(stereo_right)){
            cl_int error_code = 0;
            OpenCLRuntime runtime;
            namespace phase_5 = mp::gpu_workloads::phase_5;
            if((error_code = phase_5::initialize(runtime)) == CL_SUCCESS){
                for(int i = 0; i < sample_count; ++i){
                    if(phase_5::pipeline(runtime, left, right, pp_dmap, downscale_factor, window_radius, min_disparity, max_disparity, threshold_value) == CL_SUCCESS && i == 0){
                        pp_dmap.save("depthmap_opencl_global.png");
                    }
                }
            }
            else{
                Profiler::add_info("ZNCC_OPENCL_GLOBAL | Failed to initialize opencl runtime: " + std::to_string(error_code));
            }
        }
    }

    //Runs the matrix addition workload
    void run_zncc_optimized_pipeline_a_workload(
        const std::string &stereo_left, const std::string &stereo_right, const int downscale_factor, const int window_radius,
         const int min_disparity, const int max_disparity, const int threshold_value, const int sample_count
    ){
        mp::Profiler::segment_start("PHASE 6 - ZNCC_OPENCL_TILED");
        mp::Image left, right, pp_dmap;
        //Load stereo images
        if(left.load_path(stereo_left) && right.load_path(stereo_right)){
            cl_int error_code = 0;
            OpenCLRuntime runtime;
            namespace phase_6 = mp::gpu_workloads::phase_6_tiled;
            if((error_code = phase_6::initialize(runtime)) == CL_SUCCESS){
                for(int i = 0; i < sample_count; ++i){
                    if(phase_6::pipeline(runtime, left, right, pp_dmap, downscale_factor, window_radius, min_disparity, max_disparity, threshold_value) == CL_SUCCESS && i == 0){
                        pp_dmap.save("depthmap_opencl_tiled.png");
                    }
                }
            }
            else{
                Profiler::add_info("ZNCC_OPENCL_TILED | Failed to initialize opencl runtime: " + std::to_string(error_code));
            }
        }
    }

    //Runs the matrix addition workload
    void run_zncc_optimized_pipeline_b_workload(
        const std::string &stereo_left, const std::string &stereo_right, const int downscale_factor, const int window_radius,
         const int min_disparity, const int max_disparity, const int threshold_value, const int sample_count
    ){
        mp::Profiler::segment_start("PHASE 6 - ZNCC_OPENCL_VECTORIZED");
        mp::Image left, right, pp_dmap;
        //Load stereo images
        if(left.load_path(stereo_left) && right.load_path(stereo_right)){
            cl_int error_code = 0;
            OpenCLRuntime runtime;
            namespace phase_6 = mp::gpu_workloads::phase_6_vectorized;
            if((error_code = phase_6::initialize(runtime)) == CL_SUCCESS){
                for(int i = 0; i < sample_count; ++i){
                    if(phase_6::pipeline(runtime, left, right, pp_dmap, downscale_factor, window_radius, min_disparity, max_disparity, threshold_value) == CL_SUCCESS && i == 0){
                        pp_dmap.save("depthmap_opencl_vectorized.png");
                    }
                }
            }
            else{
                Profiler::add_info("ZNCC_OPENCL_VECTORIZED | Failed to initialize opencl runtime: " + std::to_string(error_code));
            }
        }
    }

    //Runs the matrix addition workload
    void run_zncc_optimized_pipeline_c_workload(
        const std::string &stereo_left, const std::string &stereo_right, const int downscale_factor, const int window_radius,
         const int min_disparity, const int max_disparity, const int threshold_value, const int sample_count
    ){
        mp::Profiler::segment_start("PHASE 6 - ZNCC_OPENCL_IMAGE2D_T");
        mp::Image left, right, pp_dmap;
        //Load stereo images
        if(left.load_path(stereo_left) && right.load_path(stereo_right)){
            cl_int error_code = 0;
            OpenCLRuntime runtime;
            namespace phase_6 = mp::gpu_workloads::phase_6_image2d_t;
            if((error_code = phase_6::initialize(runtime)) == CL_SUCCESS){
                for(int i = 0; i < sample_count; ++i){
                    if(phase_6::pipeline(runtime, left, right, pp_dmap, downscale_factor, window_radius, min_disparity, max_disparity, threshold_value) == CL_SUCCESS && i == 0){
                        pp_dmap.save("depthmap_opencl_image2d_t.png");
                    }
                }
            }
            else{
                Profiler::add_info("ZNCC_OPENCL_IMAGE2D_T | Failed to initialize opencl runtime: " + std::to_string(error_code));
            }
        }
    }

    //Runs the matrix addition workload
    void run_zncc_optimized_integral_pipeline_workload(
        const std::string &stereo_left, const std::string &stereo_right, const int downscale_factor, const int window_radius,
         const int min_disparity, const int max_disparity, const int threshold_value, const int sample_count
    ){
        mp::Profiler::segment_start("PHASE 7 - ZNCC_OPENCL_INTEGRAL");
        mp::Image left, right, pp_dmap;
        //Load stereo images
        if(left.load_path(stereo_left) && right.load_path(stereo_right)){
            cl_int error_code = 0;
            OpenCLRuntime runtime;
            namespace phase_7 = mp::gpu_workloads::phase_7;
            if((error_code = phase_7::initialize(runtime)) == CL_SUCCESS){
                for(int i = 0; i < sample_count; ++i){
                    if(phase_7::pipeline(runtime, left, right, pp_dmap, downscale_factor, window_radius, min_disparity, max_disparity, threshold_value) == CL_SUCCESS && i == 0){
                        pp_dmap.save("depthmap_opencl_integral.png");
                    }
                }
            }
            else{
                Profiler::add_info("ZNCC_OPENCL_INTEGRAL | Failed to initialize opencl runtime: " + std::to_string(error_code));
            }
        }
    }

}