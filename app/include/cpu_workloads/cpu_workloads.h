#ifndef MP_COURSE_CPU_WORKLOADS_CPU_WORKLOADS_H
#define MP_COURSE_CPU_WORKLOADS_CPU_WORKLOADS_H

#include <string>
#include <thread_pool.h>

namespace mp::cpu_workloads{

    //Runs the matrix addition workload
    void run_matrix_addtion_workload(float * m1, float * m2, const int matrix_size, const int sample_count);

    //Run the zncc in a single thread enviroment
    void run_zncc_single_thread_workload(    
        const std::string& stereo_left, const std::string& stereo_right, const int resize_factor, const int window_size,
        const int min_disparity, const int max_disparity, const int cross_check_threshold, const int sample_count
    );

    //Run the zncc in a single thread enviroment
    void run_zncc_single_thread_workload_vectorized(    
        const std::string& stereo_left, const std::string& stereo_right, const int resize_factor, const int window_size,
        const int min_disparity, const int max_disparity, const int cross_check_threshold, const int sample_count
    );

    //Run the zncc in a multi thread enviroment workload
    void run_zncc_multi_thread_workload(
        const std::string& stereo_left, const std::string& stereo_right, const int resize_factor, const int window_size,
        const int min_disparity, const int max_disparity, const int cross_check_threshold, const int sample_count, ThreadPool& thread_pool
    );

    //Run the zncc in a multi thread enviroment workload
    void run_zncc_multi_thread_workload_vectorized(
        const std::string& stereo_left, const std::string& stereo_right, const int resize_factor, const int window_size,
        const int min_disparity, const int max_disparity, const int cross_check_threshold, const int sample_count, ThreadPool& thread_pool
    );

    //Run the zncc in a multi thread enviroment workload
    void run_zncc_multi_thread_workload_integral(
        const std::string& stereo_left, const std::string& stereo_right, const int resize_factor, const int window_size,
        const int min_disparity, const int max_disparity, const int cross_check_threshold, const int sample_count, ThreadPool& thread_pool
    );
}

#endif