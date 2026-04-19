#ifndef MP_COURSE_OPENCL_WORKLOADS_PHASE_2_H
#define MP_COURSE_OPENCL_WORKLOADS_PHASE_2_H

#include <opencl_runtime.h>

namespace mp_course::gpu_workloads::phase_2{
    //Lists the opencl information of the computer
    void list_info();

    //Load the opencl kernel from a file
    bool initialize(OpenCLRuntime& runtime);

    //Runs the matrix add kernel on the given matrices
    cl_int add_matrix_pipeline(OpenCLRuntime& runtime, float * m1, float * m2, const int matrix_size);
}

#endif