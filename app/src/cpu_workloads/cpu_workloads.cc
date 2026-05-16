#include <profiler.h>
#include <cpu_workloads/phase_2.h>
#include <cpu_workloads/phase_3.h>
#include <cpu_workloads/phase_3_vectorized.h>
#include <cpu_workloads/phase_4.h>
#include <cpu_workloads/phase_4_vectorized.h>
#include <cpu_workloads/phase_7.h>
#include <cpu_workloads/cpu_workloads.h>

namespace mp::cpu_workloads{

    //Run the zncc in a single thread enviroment workload
    void run_zncc_single_thread_workload(
        const std::string& stereo_left, const std::string& stereo_right, const int resize_factor, const int window_size,
        const int min_disparity, const int max_disparity, const int cross_check_threshold, const int sample_count
    ){
        mp::Profiler::segment_start("PHASE 3 - ZNCC_SINGLETHREAD");
        namespace phase_3 = mp::cpu_workloads::phase_3;
        mp::Image left, right, lscaled, rscaled, dmap_left, dmap_right, pp_dmap;
        //Load stereo images
        if(left.load_path(stereo_left) && right.load_path(stereo_right)){
            //Resize and grayscale the stereo images
            for(int i = 0; i < sample_count; ++i){
                if(
                    phase_3::resize_image(left, lscaled, resize_factor) && phase_3::resize_image(right, rscaled, resize_factor) &&
                    phase_3::grayscale_image(lscaled) && phase_3::grayscale_image(rscaled)
                ){
                    //Save the preprocessed images. Calculate Dmaps from them, post process the maps
                    //lscaled.save("single_preprocessed_left.png"); rscaled.save("single_preprocessed_right.png");
                    if(
                        phase_3::calculate_disparity_map(window_size, min_disparity, max_disparity, true, lscaled, rscaled, dmap_left, "left") &&
                        phase_3::calculate_disparity_map(window_size, min_disparity, max_disparity, false, rscaled, lscaled, dmap_right, "right") &&
                        phase_3::cross_check_occulsion_disparity_maps(cross_check_threshold, window_size, min_disparity, max_disparity, dmap_left, dmap_right, pp_dmap) &&
                        i == 0
                    ){
                        //Save the disparity maps
                        //dmap_left.save("left_disparity_map.png"); dmap_right.save("right_disparity_map.png");
                        pp_dmap.save("depthmap_cpu_singlethread.png");
                    }
                }
            }
        }
    }

    //Run the zncc in a single thread enviroment workload
    void run_zncc_single_thread_workload_vectorized(
        const std::string& stereo_left, const std::string& stereo_right, const int resize_factor, const int window_size,
        const int min_disparity, const int max_disparity, const int cross_check_threshold, const int sample_count
    ){
        mp::Profiler::segment_start("PHASE 3 - ZNCC_SINGLETHREAD_VECTORIZED");
        namespace phase_3 = mp::cpu_workloads::phase_3_vectorized;
        mp::Image left, right, lscaled, rscaled, dmap_left, dmap_right, pp_dmap;
        //Load stereo images
        if(left.load_path(stereo_left) && right.load_path(stereo_right)){
            //Resize and grayscale the stereo images
            for(int i = 0; i < sample_count; ++i){
                if(
                    phase_3::resize_image(left, lscaled, resize_factor) && phase_3::resize_image(right, rscaled, resize_factor) &&
                    phase_3::grayscale_image(lscaled) && phase_3::grayscale_image(rscaled)
                ){
                    //Save the preprocessed images. Calculate Dmaps from them, post process the maps
                    //lscaled.save("single_preprocessed_left.png"); rscaled.save("single_preprocessed_right.png");
                    if(
                        phase_3::calculate_disparity_map(window_size, min_disparity, max_disparity, true, lscaled, rscaled, dmap_left, "left") &&
                        phase_3::calculate_disparity_map(window_size, min_disparity, max_disparity, false, rscaled, lscaled, dmap_right, "right") &&
                        phase_3::cross_check_occulsion_disparity_maps(cross_check_threshold, window_size, min_disparity, max_disparity, dmap_left, dmap_right, pp_dmap) &&
                        i == 0
                    ){
                        //Save the disparity maps
                        //dmap_left.save("left_disparity_map.png"); dmap_right.save("right_disparity_map.png");
                        pp_dmap.save("depthmap_cpu_singlethread_vectorized.png");
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
        mp::Profiler::segment_start("PHASE 4 - ZNCC_MULTITHREAD");
        namespace phase_4 = mp::cpu_workloads::phase_4;
        mp::Image left, right, lscaled, rscaled, dmap_left, dmap_right, pp_dmap;
        //Load stereo images
        if(left.load_path(stereo_left) && right.load_path(stereo_right)){
            for(int i = 0; i < sample_count; ++i){
                //Resize and grayscale the stereo images
                if(
                    phase_4::resize_image(left, lscaled, resize_factor, thread_pool) && phase_4::resize_image(right, rscaled, resize_factor, thread_pool) &&
                    phase_4::grayscale_image(lscaled, thread_pool) && phase_4::grayscale_image(rscaled, thread_pool)
                ){
                    //Save the preprocessed images. Calculate Dmaps from them, post process the maps
                    //lscaled.save("multi_preprocessed_left.png"); rscaled.save("multi_preprocessed_right.png");
                    if(
                        phase_4::calculate_disparity_map(window_size, min_disparity, max_disparity, true, lscaled, rscaled, dmap_left, thread_pool, "left") &&
                        phase_4::calculate_disparity_map(window_size, min_disparity, max_disparity, false, rscaled, lscaled, dmap_right, thread_pool, "right") &&
                        phase_4::cross_check_occulsion_disparity_maps(cross_check_threshold, window_size, min_disparity, max_disparity, dmap_left, dmap_right, pp_dmap, thread_pool) &&
                        i == 0
                    ){
                        //Save the disparity maps
                        //dmap_left.save("left_disparity_map.png"); dmap_right.save("right_disparity_map.png");
                        pp_dmap.save("depthmap_cpu_multithreaded.png");
                    }
                }
            }
        }
    }

    //Run the zncc in a multi thread enviroment workload
    void run_zncc_multi_thread_workload_vectorized(
        const std::string& stereo_left, const std::string& stereo_right, const int resize_factor, const int window_size,
        const int min_disparity, const int max_disparity, const int cross_check_threshold, const int sample_count, ThreadPool& thread_pool
    ){
        mp::Profiler::segment_start("PHASE 4 - ZNCC_MULTITHREAD_VECTORIZED");
        namespace phase_4 = mp::cpu_workloads::phase_4_vectorized;
        mp::Image left, right, lscaled, rscaled, dmap_left, dmap_right, pp_dmap;
        //Load stereo images
        if(left.load_path(stereo_left) && right.load_path(stereo_right)){
            for(int i = 0; i < sample_count; ++i){
                //Resize and grayscale the stereo images
                if(
                    phase_4::resize_image(left, lscaled, resize_factor, thread_pool) && phase_4::resize_image(right, rscaled, resize_factor, thread_pool) &&
                    phase_4::grayscale_image(lscaled, thread_pool) && phase_4::grayscale_image(rscaled, thread_pool)
                ){
                    //Save the preprocessed images. Calculate Dmaps from them, post process the maps
                    //lscaled.save("multi_preprocessed_left.png"); rscaled.save("multi_preprocessed_right.png");
                    if(
                        phase_4::calculate_disparity_map(window_size, min_disparity, max_disparity, true, lscaled, rscaled, dmap_left, thread_pool, "left") &&
                        phase_4::calculate_disparity_map(window_size, min_disparity, max_disparity, false, rscaled, lscaled, dmap_right, thread_pool, "right") &&
                        phase_4::cross_check_occulsion_disparity_maps(cross_check_threshold, window_size, min_disparity, max_disparity, dmap_left, dmap_right, pp_dmap, thread_pool) &&
                        i == 0
                    ){
                        //Save the disparity maps
                        //dmap_left.save("left_disparity_map.png"); dmap_right.save("right_disparity_map.png");
                        pp_dmap.save("depthmap_cpu_multithreaded_vectorized.png");
                    }
                }
            }
        }
    }

    //Run the zncc in a multi thread enviroment workload
    void run_zncc_multi_thread_workload_integral(
        const std::string& stereo_left, const std::string& stereo_right, const int resize_factor, const int window_size,
        const int min_disparity, const int max_disparity, const int cross_check_threshold, const int sample_count, ThreadPool& thread_pool
    ){
        mp::Profiler::segment_start("PHASE 7 - ZNCC_MULTITHREAD_INTEGRAL");
        namespace phase_7 = mp::cpu_workloads::phase_7;
        mp::Image ls_map, rs_map, lsq_map, rsq_map;
        mp::Image left, right, lscaled, rscaled, dmap_left, dmap_right, pp_dmap;
        //Load stereo images
        if(left.load_path(stereo_left) && right.load_path(stereo_right)){
            for(int i = 0; i < sample_count; ++i){
                //Resize and grayscale the stereo images
                if(
                    phase_7::resize_image(left, lscaled, resize_factor, thread_pool) && phase_7::resize_image(right, rscaled, resize_factor, thread_pool) &&
                    phase_7::grayscale_image(lscaled, thread_pool) && phase_7::grayscale_image(rscaled, thread_pool)
                ){
                    //Save the preprocessed images. Calculate Dmaps from them, post process the maps
                    //lscaled.save("multi_preprocessed_left.png"); right.save("multi_preprocessed_right.png");
                    if(
                        phase_7::calculate_integral_map(lscaled, ls_map, lsq_map, window_size, max_disparity, false, thread_pool) &&
                        phase_7::calculate_integral_map(rscaled, rs_map, rsq_map, window_size, max_disparity, true, thread_pool) &&
                        phase_7::calculate_disparity_map(
                            window_size, min_disparity, max_disparity, true,
                            ls_map, rs_map, lsq_map, rsq_map,
                            lscaled, rscaled, dmap_left, thread_pool, "left"
                        ) &&
                        phase_7::calculate_disparity_map(
                            window_size, min_disparity, max_disparity, false,
                            rs_map, ls_map, rsq_map, lsq_map,
                            rscaled, lscaled, dmap_right, thread_pool, "right"
                        ) &&
                        phase_7::cross_check_occulsion_disparity_maps(cross_check_threshold, window_size, min_disparity, max_disparity, dmap_left, dmap_right, pp_dmap, thread_pool) &&
                        i == 0
                    ){
                        //Save the disparity maps
                        /*ls_map.save("ls_map.png"); rs_map.save("rs_map.png");
                        lsq_map.save("lsq_map.png"); rsq_map.save("rsq_map.png");
                        dmap_left.save("left_disparity_map.png"); dmap_right.save("right_disparity_map.png");*/
                        pp_dmap.save("depthmap_cpu_multithreaded_integral.png");
                    }
                }
            }
        }
    }

    //Runs the matrix addition workload
    void run_matrix_addtion_workload(std::vector<float>& m1, std::vector<float>& m2, const int matrix_size, const int sample_count){
        namespace phase_2 = mp::cpu_workloads::phase_2;
        mp::Profiler::segment_start("PHASE 2 - cpu_add_matrix");
        //Run the sum calculation
        for (int i = 0; i < sample_count; ++i){
            phase_2::add_matrix(m1, m2, matrix_size);
        }
    }

}