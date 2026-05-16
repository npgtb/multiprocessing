#ifndef MP_COURSE_COMBINED_WORKLOADS_H
#define MP_COURSE_COMBINED_WORKLOADS_H

#include <string>
#include <thread_pool.h>

namespace mp{

    //Generates matrices and runs the cpu and gpu versions of the addition
    void run_combined_add_matrix_workload(const int sample_count);

    //Runs the cpu versions and gpu version of the zncc algo
    void run_combined_zncc_workloads(
        const std::string& stereo_left, const std::string& stereo_right, const int resize_factor, const int window_size,
        const int min_disparity, const int max_disparity, const int cross_check_threshold, const int sample_count, 
        mp::ThreadPool& thread_pool
    );

}

#endif