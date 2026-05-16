#include <combined_workloads.h>
#include <cpu_workloads/cpu_workloads.h>
#include <opencl_workloads/opencl_workloads.h>

namespace mp{
    //Generates matrices and runs the cpu and gpu versions of the addition
    void run_combined_add_matrix_workload(const int sample_count){
        //Generate work
        constexpr int size = 100;
        std::vector<float> m1(size), m2(size);
        for(int i = 0; i < size; ++i){
            m1[i] = i;
            m2[i] = i;
        }
        //Run the cpu and gpu versions
        cpu_workloads::run_matrix_addtion_workload(m1, m2, size, sample_count);
        gpu_workloads::run_matrix_addtion_workload(m1, m2, size, sample_count);
    }

    //Runs the cpu versions and gpu version of the zncc algo
    void run_combined_zncc_workloads(
        const std::string& stereo_left, const std::string& stereo_right, const int resize_factor, const int window_size,
        const int min_disparity, const int max_disparity, const int cross_check_threshold, const int sample_count,
        mp::ThreadPool& thread_pool
    ){
        //Sample the runtimes of the zncc implementations
        cpu_workloads::run_zncc_single_thread_workload(stereo_left, stereo_right, resize_factor, window_size, min_disparity, max_disparity, cross_check_threshold, sample_count);
        cpu_workloads::run_zncc_single_thread_workload_vectorized(stereo_left, stereo_right, resize_factor, window_size, min_disparity, max_disparity, cross_check_threshold, sample_count);
        cpu_workloads::run_zncc_multi_thread_workload(stereo_left, stereo_right, resize_factor, window_size, min_disparity, max_disparity, cross_check_threshold, sample_count, thread_pool);
        cpu_workloads::run_zncc_multi_thread_workload_vectorized(stereo_left, stereo_right, resize_factor, window_size, min_disparity, max_disparity, cross_check_threshold, sample_count, thread_pool);
        cpu_workloads::run_zncc_multi_thread_workload_integral(stereo_left, stereo_right, resize_factor, window_size, min_disparity, max_disparity, cross_check_threshold, sample_count, thread_pool);
        gpu_workloads::run_zncc_pipeline_global_workload(stereo_left, stereo_right, resize_factor, window_size, min_disparity, max_disparity, cross_check_threshold, sample_count);
        gpu_workloads::run_zncc_pipeline_tiled_workload(stereo_left, stereo_right, resize_factor, window_size, min_disparity, max_disparity, cross_check_threshold, sample_count);
        gpu_workloads::run_zncc_pipeline_vectorized_workload(stereo_left, stereo_right, resize_factor, window_size, min_disparity, max_disparity, cross_check_threshold, sample_count);
        gpu_workloads::run_zncc_pipeline_image2d_workload(stereo_left, stereo_right, resize_factor, window_size, min_disparity, max_disparity, cross_check_threshold, sample_count);
        gpu_workloads::run_zncc_optimized_integral_pipeline_workload(stereo_left, stereo_right, resize_factor, window_size, min_disparity, max_disparity, cross_check_threshold, sample_count);
        gpu_workloads::run_zncc_optimized_integral_aps_pipeline_workload(stereo_left, stereo_right, resize_factor, window_size, min_disparity, max_disparity, cross_check_threshold, sample_count);
        gpu_workloads::run_zncc_optimized_integral_aps_cl_pipeline_workload(stereo_left, stereo_right, resize_factor, window_size, min_disparity, max_disparity, cross_check_threshold, sample_count);
    }
}