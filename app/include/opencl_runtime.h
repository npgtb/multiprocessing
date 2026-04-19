#ifndef MP_COURSE_OPENCL_PROGRAM_H
#define MP_COURSE_OPENCL_PROGRAM_H

#include <string>
#include <vector>
#include <opencl_wrapper/opencl_wrapper.h>

namespace mp_course{
    struct OpenCLRuntime{
        std::shared_ptr<clw::Context> context;
        std::shared_ptr<clw::Program> program;
        std::vector<std::shared_ptr<clw::Kernel>> kernels;
        std::shared_ptr<clw::CommandQueue> cc_queue;

        //Init empty
        OpenCLRuntime();

        //Try to load from file
        cl_int load_file(clw::Device& device, const std::string& path, const std::vector<std::string>& kernel_names);
    };
}

#endif