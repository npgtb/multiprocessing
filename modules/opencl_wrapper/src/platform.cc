#include <opencl_wrapper/platform.h>

namespace clw{

    //Initialize empty platform
    Platform::Platform():id(nullptr){}

    //Initialize platform with id
    Platform::Platform(cl_platform_id platform_id):id(platform_id){}

    //Copy constructors
    Platform::Platform(Platform && other){
        id = other.id;
    }

    //Copy constructors
    Platform::Platform(const Platform& other){
        id = other.id;
    }

    //Assignment operators
    Platform& Platform::operator=(Platform&& other){
        id = other.id;
        return *this;
    }
    
    //Assignment operators
    Platform& Platform::operator=(const Platform& other){
        id = other.id;
        return *this;
    }

    //Retrieve the data size of the cl_device_info
    ErrorOr<size_t> Platform::info_size(cl_platform_info info_id){
        size_t info_size = 0;
        cl_int error_code = clGetPlatformInfo(id, info_id, 0, nullptr, &info_size);
        if(error_code == CL_SUCCESS){
            return info_size;
        }
        return error_code;
    }

    //Retrieve cl_device_info
    cl_int Platform::info(void* info, size_t size, cl_platform_info info_id){
        return clGetPlatformInfo(id, info_id, size, info, nullptr);
    }

    //Retrieves the available devices of the platform
    ErrorOr<std::vector<Device>> Platform::devices(){
        cl_uint device_count = 0;
        //https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetDeviceIDs.html
        //Query the device count: id, device_type, buffer size, IGNORED, &query_result
        cl_int error_code = clGetDeviceIDs(id, CL_DEVICE_TYPE_ALL, 0, NULL, &device_count);
        if(error_code == CL_SUCCESS){
            std::vector<cl_device_id> device_ids(device_count);
            //Query the device information: id, device_type, buffer size, buffer, IGNORED
            error_code = clGetDeviceIDs(id, CL_DEVICE_TYPE_ALL, device_count, device_ids.data(), nullptr);
            if(error_code == CL_SUCCESS){
                std::vector<Device> devices;
                devices.reserve(device_count);
                for(auto device_id : device_ids){
                    devices.emplace_back(device_id);
                }
                return devices;
            }
        }
        return error_code;
    }

    //Queries all the available platforms in the system
    ErrorOr<std::vector<Platform>> Platform::available(){
        cl_uint platform_count = 0;
        //https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetPlatformIDs.html
        //Query the available platforms opencl implementations: buffer size, buffer, &query_result
        cl_int error_code = clGetPlatformIDs(0, NULL, &platform_count);
        if(error_code == CL_SUCCESS){
            //Fill the array with n copies
            std::vector<cl_platform_id> platform_ids(platform_count);
            //buffer size, buffer, IGNORED
            error_code = clGetPlatformIDs(platform_count, platform_ids.data(), nullptr);
            if(error_code == CL_SUCCESS){
                std::vector<Platform> platforms;
                //Reserve space
                platforms.reserve(platform_count);
                for(auto platform_id : platform_ids){
                    platforms.emplace_back(platform_id);
                }
                return platforms;
            }
        }
        return error_code;
    }

}