#include <profiler.h>
#include <cpu_workloads/phase_2.h>
#include <cpu_workloads/phase_3.h>
#include <cpu_workloads/phase_4.h>
#include <cpu_workloads/cpu_workloads.h>

namespace mp_course::cpu_workloads{

    //Run the zncc in a single thread enviroment workload
    void run_zncc_single_thread_workload(
        const std::string& stereo_left, const std::string& stereo_right, const int resize_factor, const int window_size,
        const int min_disparity, const int max_disparity, const int cross_check_threshold, const int sample_count
    ){
        mp_course::Profiler::segment_start("PHASE 3 - run_zncc_single_thread_workload");
        namespace phase_3 = mp_course::cpu_workloads::phase_3;
        mp_course::Image left, right, dmap_left, dmap_right, pp_dmap;
        //Load stereo images
        if(left.load_path(stereo_left) && right.load_path(stereo_right)){
            //Resize and grayscale the stereo images
            for(int i = 0; i < sample_count; ++i){
                if(
                    phase_3::resize_image(left, resize_factor) && phase_3::resize_image(right, resize_factor) &&
                    phase_3::grayscale_image(left) && phase_3::grayscale_image(right)
                ){
                    //Save the preprocessed images. Calculate Dmaps from them, post process the maps
                    //left.save("single_preprocessed_left.png"); right.save("single_preprocessed_right.png");
                    if(
                        phase_3::calculate_disparity_map(window_size, min_disparity, max_disparity, left, right, dmap_left) &&
                        phase_3::calculate_disparity_map(window_size, min_disparity, max_disparity, right, left, dmap_right) &&
                        phase_3::cross_check_occulsion_disparity_maps(cross_check_threshold, window_size, min_disparity, max_disparity, dmap_left, dmap_right, pp_dmap) &&
                        i == 0
                    ){
                        //Save the disparity maps
                        //dmap_left.save("left_disparity_map.png"); dmap_right.save("right_disparity_map.png");
                        pp_dmap.save("single_depthmap.png");
                    }
                }
            }
        }
    }

    //Run the zncc in a multi thread enviroment workload
    void run_zncc_multi_thread_workload(
        const std::string& stereo_left, const std::string& stereo_right, const int resize_factor, const int window_size,
        const int min_disparity, const int max_disparity, const int cross_check_threshold, const int sample_count, ThreadPool& thread_pool
    ){
        mp_course::Profiler::segment_start("PHASE 4 - run_zncc_multi_thread_workload");
        namespace phase_4 = mp_course::cpu_workloads::phase_4;
        mp_course::Image left, right, dmap_left, dmap_right, pp_dmap;
        //Load stereo images
        if(left.load_path(stereo_left) && right.load_path(stereo_right)){
            for(int i = 0; i < sample_count; ++i){
                //Resize and grayscale the stereo images
                if(
                    phase_4::resize_image(left, resize_factor, thread_pool) && phase_4::resize_image(right, resize_factor, thread_pool) &&
                    phase_4::grayscale_image(left, thread_pool) && phase_4::grayscale_image(right, thread_pool)
                ){
                    //Save the preprocessed images. Calculate Dmaps from them, post process the maps
                    //left.save("multi_preprocessed_left.png"); right.save("multi_preprocessed_right.png");
                    if(
                        phase_4::calculate_disparity_map(window_size, min_disparity, max_disparity, left, right, dmap_left, thread_pool) &&
                        phase_4::calculate_disparity_map(window_size, min_disparity, max_disparity, right, left, dmap_right, thread_pool) &&
                        phase_4::cross_check_occulsion_disparity_maps(cross_check_threshold, window_size, min_disparity, max_disparity, dmap_left, dmap_right, pp_dmap, thread_pool) &&
                        i == 0
                    ){
                        //Save the disparity maps
                        //dmap_left.save("left_disparity_map.png"); dmap_right.save("right_disparity_map.png");
                        pp_dmap.save("multi_depthmap.png");
                    }
                }
            }
        }
    }

    //Runs the matrix addition workload
    void run_matrix_addtion_workload(float * m1, float * m2, const int matrix_size, const int sample_count){
        namespace phase_2 = mp_course::cpu_workloads::phase_2;
        mp_course::Profiler::segment_start("PHASE 2 - cpu_add_matrix");
        //Run the sum calculation
        for (int i = 0; i < sample_count; ++i){
            phase_2::add_matrix(m1, m2, matrix_size);
        }
    }

}