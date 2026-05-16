#ifndef MP_COURSE_CPU_WORKLOADS_PHASE_2_H
#define MP_COURSE_CPU_WORKLOADS_PHASE_2_H

#include <vector>

namespace mp::cpu_workloads::phase_2{
    //Performs a element wise addition on two float arrays(Matrices)
    void add_matrix(std::vector<float>& m1, std::vector<float>& m2, const int size);
}

#endif