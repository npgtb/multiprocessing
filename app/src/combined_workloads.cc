#include <combined_workloads.h>
#include <cpu_workloads/cpu_workloads.h>
#include <opencl_workloads/opencl_workloads.h>

namespace mp_course{
    //Generates matrices and runs the cpu and gpu versions of the addition
    void run_combined_add_matrix_workload(const int sample_count){
        //Generate work
        constexpr int size = 100;
        float * m1 = static_cast<float*>(malloc(size * sizeof(float)));
        float * m2 = static_cast<float*>(malloc(size * sizeof(float)));
        for(int i = 0; i < size; ++i){
            m1[i] = i;
            m2[i] = i;
        }
        //Run the cpu and gpu versions
        cpu_workloads::run_matrix_addtion_workload(m1, m2, size, sample_count);
        gpu_workloads::run_matrix_addtion_workload(m1, m2, size, sample_count);
        free(m1); free(m2);
    }

    //Runs the cpu versions and gpu version of the zncc algo
    void run_combined_zncc_workloads(
        const std::string& stereo_left, const std::string& stereo_right, const int resize_factor, const int window_size,
        const int min_disparity, const int max_disparity, const int cross_check_threshold, const int sample_count,
        mp_course::ThreadPool& thread_pool
    ){
        cpu_workloads::run_zncc_single_thread_workload(stereo_left, stereo_right, resize_factor, window_size, min_disparity, max_disparity, cross_check_threshold, sample_count);
        cpu_workloads::run_zncc_multi_thread_workload(stereo_left, stereo_right, resize_factor, window_size, min_disparity, max_disparity, cross_check_threshold, sample_count, thread_pool);
        gpu_workloads::run_zncc_pipeline_workload(stereo_left, stereo_right, resize_factor, window_size, min_disparity, max_disparity, cross_check_threshold, sample_count);
        gpu_workloads::run_zncc_optimized_pipeline_a_workload(stereo_left, stereo_right, resize_factor, window_size, min_disparity, max_disparity, cross_check_threshold, sample_count);
        gpu_workloads::run_zncc_optimized_pipeline_b_workload(stereo_left, stereo_right, resize_factor, window_size, min_disparity, max_disparity, cross_check_threshold, sample_count);
        gpu_workloads::run_zncc_optimized_pipeline_c_workload(stereo_left, stereo_right, resize_factor, window_size, min_disparity, max_disparity, cross_check_threshold, sample_count);
    }
}