#ifndef MP_COURSE_OPENCL_WORKLOADS_PHASE_6_C
#define MP_COURSE_OPENCL_WORKLOADS_PHASE_6_C

#include <image.h>
#include <opencl_runtime.h>

namespace mp_course::gpu_workloads::phase_6_image2d_t{

    //Loads the opencl file and initializes the given kernels from it
    cl_int initialize(OpenCLRuntime& runtime);

    //Split pipeline rescale => grayscale, combine => Zncc
    cl_int pipeline(
        OpenCLRuntime& runtime, Image& left, Image& right, Image& map,
        const int downscale_factor, const int window_radius, const int min_disparity,
        const int max_disparity, const int threshold_value
    );
}

#endif