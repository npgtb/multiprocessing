#ifndef MP_COURSE_OPENCL_WORKLOADS_PHASE_5
#define MP_COURSE_OPENCL_WORKLOADS_PHASE_5

#include <image.h>
#include <opencl_runtime.h>

namespace mp_course::gpu_workloads::phase_5{

    //Loads the opencl file and initializes the given kernels from it
    bool initialize(OpenCLRuntime& runtime);

    //Split pipeline rescale => grayscale, combine => Zncc
    cl_int pipeline(
        OpenCLRuntime& runtime, Image& left, Image& right, Image& map,
        const int downscale_factor, const int window_radius, const int min_disparity,
        const int max_disparity, const int threshold_value
    );
}

#endif