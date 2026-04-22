#ifndef MP_COURSE_CPU_WORKLOADS_PHASE_4_H
#define MP_COURSE_CPU_WORKLOADS_PHASE_4_H

#include <string>
#include <image.h>
#include <thread_pool.h>

namespace mp_course::cpu_workloads::phase_4{

    //Resize the image by factor. Takes the nth row and column approach. Expects a RGBA format image.
    bool resize_image(Image& image, const int factor, ThreadPool& thread_pool);

    //Grayscales the image. Expects a RGBA format image and turns image to GRAY format.
    bool grayscale_image(Image& image, ThreadPool& thread_pool);

    //Calculates the disparity using the ZNCC algo. Calculates disparity shift from left image to right image, storing values in map image.
    bool calculate_disparity_map(int window_radius, int min_disparity, int max_disparity, const bool left_to_right, Image& left, Image& right, Image& map, ThreadPool& thread_pool, std::string scope_tag);

    //Cross-checks the two disparity maps against each other
    bool cross_check_occulsion_disparity_maps(int threshold_value, int window_radius, int min_disparity, int max_disparity,Image& left_disparity, Image& right_disparity, Image& pp_disparity, ThreadPool& thread_pool);
}


#endif