#ifndef OPENCL_WRAPPER_DEVICE_H
#define OPENCL_WRAPPER_DEVICE_H

#include <memory>
#include <vector>
#include <opencl_wrapper/opencl.h>
#include <opencl_wrapper/context.h>
#include <opencl_wrapper/error_or.h>


namespace clw{
    struct Device{
        cl_device_id id;

        //Initialize empty Device
        Device();

        //Initialize Device with id
        Device(cl_device_id device_id);

        //Copy constructors
        Device(Device && other);
        Device(const Device&);

        //Assignment operators
        Device& operator=(Device&& other);
        Device& operator=(const Device&);

        //Retrieve the data size of the cl_device_info
        ErrorOr<size_t> info_size(cl_device_info info_id);

        //Retrieve cl_device_info
        cl_int info(void* info, size_t size, cl_device_info info_id);

        //Create a context object
        static ErrorOr<std::shared_ptr<Context>> create_context(const std::vector<Device>& context_devices);

    };
}

#endif