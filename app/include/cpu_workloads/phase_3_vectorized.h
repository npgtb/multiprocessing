#ifndef MP_COURSE_CPU_WORKLOADS_PHASE_3_VECTORIZED_H
#define MP_COURSE_CPU_WORKLOADS_PHASE_3_VECTORIZED_H

#include <string>
#include <image.h>


namespace mp_course::cpu_workloads::phase_3_vectorized{

    //Resize the image by factor. Takes the nth row and column approach. Expects a RGBA format image.
    bool resize_image(Image& image, const int factor);

    //Grayscales the image. Expects a RGBA format image and turns image to GRAY format.
    bool grayscale_image(Image& image);

    //Calculates the disparity using the ZNCC algo. Calculates disparity shift from left image to right image, storing values in map image.
    bool calculate_disparity_map(const int window_radius, const int min_disparity, const int max_disparity, const bool left_to_right, Image& left, Image& right, Image& map, std::string scope_tag);

    //Cross-checks the two disparity maps against each other
    bool cross_check_occulsion_disparity_maps(const int threshold_value, const int window_radius, const int min_disparity, const int max_disparity, Image& left_disparity, Image& right_disparity, Image& pp_disparity);
}

#endif