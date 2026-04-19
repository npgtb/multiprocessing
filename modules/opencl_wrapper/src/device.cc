#include "opencl_wrapper/context.h"
#include <opencl_wrapper/device.h>

namespace clw{
    //Initialize empty Device
    Device::Device():id(nullptr){}

    //Initialize Device with id
    Device::Device(cl_device_id device_id):id(device_id){}

    //Copy constructors
    Device::Device(Device && other){
        id = other.id;
    }

    //Copy constructors
    Device::Device(const Device& other){
        id = other.id;
    }

    //Assignment operators
    Device& Device::operator=(Device&& other){
        id = other.id;
        return *this;
    }
    
    //Assignment operators
    Device& Device::operator=(const Device& other){
        id = other.id;
        return *this;
    }

    //Retrieve the data size of the cl_device_info
    ErrorOr<size_t> Device::info_size(cl_device_info info_id){
        size_t info_size = 0;
        cl_int error_code = clGetDeviceInfo(id, info_id, 0, nullptr, &info_size);
        if(error_code == CL_SUCCESS){
            return info_size;
        }
        return error_code;
    }

    //Retrieve cl_device_info
    cl_int Device::info(void* info, size_t size, cl_device_info info_id){
        return clGetDeviceInfo(id, info_id, size, info, nullptr);
    }

    //Create a context object
    ErrorOr<std::shared_ptr<Context>> Device::create_context(const std::vector<Device>& context_devices){
        cl_int error_code = 0;
        std::vector<cl_device_id> device_ids;
        device_ids.reserve(context_devices.size());
        for(const auto& device: context_devices){
            device_ids.push_back(device.id);
        }
        //https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clCreateContext.html
        //Create a context from the device: ctxt properties, number of devices, device list, report back func call, data to report back, error code
        std::shared_ptr<Context> context = std::make_shared<Context>(
            clCreateContext(NULL, device_ids.size(), device_ids.data(), NULL, NULL, &error_code)
        );
        if(error_code == CL_SUCCESS){
            return context;
        }
        return error_code;
    }

}