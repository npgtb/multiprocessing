#ifndef MP_COURSE_OPENCL_WORKLOADS_OPENCL_WORKLOADS_H
#define MP_COURSE_OPENCL_WORKLOADS_OPENCL_WORKLOADS_H

#include <image.h>

namespace mp::gpu_workloads{
    //Runs the opencl hello world program
    void run_hello_world_workload();

    //Runs the list info workload
    void run_list_info_workload();

    //Runs the matrix addition workload
    void run_matrix_addtion_workload(std::vector<float>& m1, std::vector<float>& m2, const int matrix_size, const int sample_count);

    //Runs the opencl Zncc Pipeline
    void run_zncc_pipeline_global_workload(
        const std::string &stereo_left, const std::string &stereo_right, const int downscale_factor, const int window_radius, 
        const int min_disparity, const int max_disparity, const int threshold_value, const int sample_count
    );

    //Runs the opencl Zncc Pipeline
    void run_zncc_pipeline_tiled_workload(
        const std::string &stereo_left, const std::string &stereo_right, const int downscale_factor, const int window_radius, 
        const int min_disparity, const int max_disparity, const int threshold_value, const int sample_count
    );

    //Runs the opencl Zncc Pipeline
    void run_zncc_pipeline_vectorized_workload(
        const std::string &stereo_left, const std::string &stereo_right, const int downscale_factor, const int window_radius, 
        const int min_disparity, const int max_disparity, const int threshold_value, const int sample_count
    );

    //Runs the opencl Zncc Pipeline
    void run_zncc_pipeline_image2d_workload(
        const std::string &stereo_left, const std::string &stereo_right, const int downscale_factor, const int window_radius, 
        const int min_disparity, const int max_disparity, const int threshold_value, const int sample_count
    );

    //Runs the opencl Zncc Pipeline
    void run_zncc_optimized_integral_pipeline_workload(
        const std::string &stereo_left, const std::string &stereo_right, const int downscale_factor, const int window_radius, 
        const int min_disparity, const int max_disparity, const int threshold_value, const int sample_count
    );

    //Runs the opencl Zncc Pipeline
    void run_zncc_optimized_integral_aps_pipeline_workload(
        const std::string &stereo_left, const std::string &stereo_right, const int downscale_factor, const int window_radius, 
        const int min_disparity, const int max_disparity, const int threshold_value, const int sample_count
    );

    //Runs the opencl Zncc Pipeline
    void run_zncc_optimized_integral_aps_cl_pipeline_workload(
        const std::string &stereo_left, const std::string &stereo_right, const int downscale_factor, const int window_radius, 
        const int min_disparity, const int max_disparity, const int threshold_value, const int sample_count
    );
}

#endif