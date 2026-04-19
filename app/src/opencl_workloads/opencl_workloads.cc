#include <profiler.h>
#include <opencl_workloads/phase_1.h>
#include <opencl_workloads/phase_2.h>
#include <opencl_workloads/phase_5.h>
#include <opencl_workloads/phase_6_a.h>
#include <opencl_workloads/phase_6_b.h>
#include <opencl_workloads/phase_6_c.h>
#include <opencl_workloads/opencl_workloads.h>

namespace mp_course::gpu_workloads{
    //Runs the opencl hello world program
    void run_hello_world_workload(){
        namespace phase_1 = mp_course::gpu_workloads::phase_1;
        mp_course::Profiler::segment_start("PHASE 1 - opencl_workload_hello_world");
        phase_1::hello_world();
    }

    //Runs the list info workload
    void run_list_info_workload(){
        namespace phase_2 = mp_course::gpu_workloads::phase_2;
        mp_course::Profiler::segment_start("PHASE 2 - opencl_workload_list_info");
        phase_2::list_info();
    }

    //Runs the matrix addition workload
    void run_matrix_addtion_workload(float * m1, float * m2, const int matrix_size, const int sample_count){
        namespace phase_2 = mp_course::gpu_workloads::phase_2;
        mp_course::Profiler::segment_start("PHASE 2 - opencl_workload_add_matrix");
        OpenCLRuntime runtime;
        if(phase_2::initialize(runtime)){
            for(int i = 0; i < sample_count; ++i){
                phase_2::add_matrix_pipeline(runtime, m1, m2, matrix_size);
            }
        }
        else{
            Profiler::add_info("Failed to initialize opencl runtime");
        }
    }

    //Runs the matrix addition workload
    void run_zncc_pipeline_workload(
        const std::string &stereo_left, const std::string &stereo_right, const int downscale_factor, const int window_radius, 
        const int min_disparity, const int max_disparity, const int threshold_value, const int sample_count
    ){
        mp_course::Profiler::segment_start("PHASE 5 - opencl_zncc_pipeline");
        mp_course::Image left, right, pp_dmap;
        //Load stereo images
        if(left.load_path(stereo_left) && right.load_path(stereo_right)){
            namespace phase_5 = mp_course::gpu_workloads::phase_5;
            OpenCLRuntime runtime;
            if(phase_5::initialize(runtime)){
                for(int i = 0; i < sample_count; ++i){
                    if(phase_5::pipeline(runtime, left, right, pp_dmap, downscale_factor, window_radius, min_disparity, max_disparity, threshold_value) == CL_SUCCESS && i == 0){
                        pp_dmap.save("opencl_depthmap.png");
                    }
                }
            }
            else{
                Profiler::add_info("Failed to initialize opencl runtime");
            }
        }
    }

    //Runs the matrix addition workload
    void run_zncc_optimized_pipeline_a_workload(
        const std::string &stereo_left, const std::string &stereo_right, const int downscale_factor, const int window_radius,
         const int min_disparity, const int max_disparity, const int threshold_value, const int sample_count
    ){
        mp_course::Profiler::segment_start("PHASE 6_Tiled - opencl_zncc_pipeline");
        mp_course::Image left, right, pp_dmap;
        //Load stereo images
        if(left.load_path(stereo_left) && right.load_path(stereo_right)){
            namespace phase_6 = mp_course::gpu_workloads::phase_6_a;
            OpenCLRuntime runtime;
            if(phase_6::initialize(runtime)){
                for(int i = 0; i < sample_count; ++i){
                    if(phase_6::pipeline(runtime, left, right, pp_dmap, downscale_factor, window_radius, min_disparity, max_disparity, threshold_value) == CL_SUCCESS && i == 0){
                        pp_dmap.save("opencl_optimized_depthmap_a.png");
                    }
                }
            }
            else{
                Profiler::add_info("Failed to initialize opencl runtime");
            }
        }
    }

    //Runs the matrix addition workload
    void run_zncc_optimized_pipeline_b_workload(
        const std::string &stereo_left, const std::string &stereo_right, const int downscale_factor, const int window_radius,
         const int min_disparity, const int max_disparity, const int threshold_value, const int sample_count
    ){
        mp_course::Profiler::segment_start("PHASE 6_Vectorized - opencl_zncc_pipeline");
        mp_course::Image left, right, pp_dmap;
        //Load stereo images
        if(left.load_path(stereo_left) && right.load_path(stereo_right)){
            namespace phase_6 = mp_course::gpu_workloads::phase_6_b;
            OpenCLRuntime runtime;
            if(phase_6::initialize(runtime)){
                for(int i = 0; i < sample_count; ++i){
                    if(phase_6::pipeline(runtime, left, right, pp_dmap, downscale_factor, window_radius, min_disparity, max_disparity, threshold_value) == CL_SUCCESS && i == 0){
                        pp_dmap.save("opencl_optimized_depthmap_b.png");
                    }
                }
            }
            else{
                Profiler::add_info("Failed to initialize opencl runtime");
            }
        }
    }

    //Runs the matrix addition workload
    void run_zncc_optimized_pipeline_c_workload(
        const std::string &stereo_left, const std::string &stereo_right, const int downscale_factor, const int window_radius,
         const int min_disparity, const int max_disparity, const int threshold_value, const int sample_count
    ){
        mp_course::Profiler::segment_start("PHASE 6_Image_2d_t - opencl_zncc_pipeline");
        mp_course::Image left, right, pp_dmap;
        //Load stereo images
        if(left.load_path(stereo_left) && right.load_path(stereo_right)){
            namespace phase_6 = mp_course::gpu_workloads::phase_6_c;
            OpenCLRuntime runtime;
            if(phase_6::initialize(runtime)){
                for(int i = 0; i < sample_count; ++i){
                    if(phase_6::pipeline(runtime, left, right, pp_dmap, downscale_factor, window_radius, min_disparity, max_disparity, threshold_value) == CL_SUCCESS && i == 0){
                        pp_dmap.save("opencl_optimized_depthmap_c.png");
                    }
                }
            }
            else{
                Profiler::add_info("Failed to initialize opencl runtime");
            }
        }
    }

}