#ifndef MP_COURSE_HELPERS_H
#define MP_COURSE_HELPERS_H

#include <vector>
#include <string>
#include <opencl_wrapper/event.h>
#include <opencl_wrapper/opencl_wrapper.h>

namespace mp{
    //Simple checksum for float array comparision
    double simple_checksum_float_array(const std::vector<float>& m);

    //Gets available platforms
    void get_platforms(std::vector<clw::Platform>& platforms);

    //Gets devices from the given platform    
    void get_devices(clw::Platform platform, std::vector<clw::Device>& devices);

    //Find a prefered computing hardware, preference order: GPU, INTEGRATED, CPU
    clw::Device prefered_device();

    //Reports the timing information of the given event to the profiler
    void report_cl_timing(const std::string& label, std::shared_ptr<clw::Event> event);

}

#endif