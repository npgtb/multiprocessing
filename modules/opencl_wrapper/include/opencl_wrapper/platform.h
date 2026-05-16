#ifndef OPENCL_WRAPPER_PLATFORM_H
#define OPENCL_WRAPPER_PLATFORM_H

#include <vector>
#include <string>
#include <opencl_wrapper/opencl.h>
#include <opencl_wrapper/device.h>
#include <opencl_wrapper/error_or.h>

namespace clw{
    
    struct Platform{
        cl_platform_id id;

        //Initialize empty platform
        Platform();

        //Initialize platform with id
        Platform(cl_platform_id platform_id);

        //Copy constructors
        Platform(Platform && other);
        Platform(const Platform&);

        //Assignment operators
        Platform& operator=(Platform&& other);
        Platform& operator=(const Platform&);

        //Retrieve the data size of the cl_device_info
        ErrorOr<size_t> info_size(cl_platform_info info_id);

        //Retrieve cl_device_info
        cl_int info(void* info, size_t size, cl_platform_info info_id);

        //Retrieves string associated with the platform
        ErrorOr<std::string> platform_string(cl_platform_info string_id);

        //Retrieves the available devices of the platform
        ErrorOr<std::vector<Device>> devices();

        //Queries all the available platforms in the system
        static ErrorOr<std::vector<Platform>> available();

    };
}

#endif