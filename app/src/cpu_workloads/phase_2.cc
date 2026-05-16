#include <helpers.h>
#include <profiler.h>
#include <scope_timer.h>
#include <cpu_workloads/phase_2.h>

namespace mp::cpu_workloads::phase_2{
    //Performs a element wise addition on two float arrays(Matrices)
    void add_matrix(std::vector<float>& m1, std::vector<float>& m2, const int size){
        ScopeTimer scope_timer("add_matrix");
        std::vector<float> summation(size);
        for(int i = 0;  i < size; ++i){
            summation[i] = m1[i] + m2[i];
        }
        Profiler::add_info("CPU result checksum: " + std::to_string(simple_checksum_float_array(summation)));
    }
}


