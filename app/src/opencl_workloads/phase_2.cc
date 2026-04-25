#include <helpers.h>
#include <functional>
#include <profiler.h>
#include <scope_timer.h>
#include <opencl_workloads/phase_2.h>

namespace mp::gpu_workloads::phase_2{

    //Define the data to fetch from the platforms and devices
    struct info_fetch{
        //Both cl_device_info and cl_platform_info are typedefs of cl_uint
        cl_uint info_id;
        std::string data_label;
        std::function<void(std::string, void*)> data_handler;
    };

    //Prints the info about the platform
    void print_platform_info(clw::Platform& platform){
        //Platform vendor, name, profile, version
        const std::vector<info_fetch> platform_info_fetches = {
            {
                CL_PLATFORM_VENDOR,
                "Platform vendor",
                [](const std::string& label, void*data_ptr){
                    if(data_ptr){
                        Profiler::add_info(label + ": " + std::string(static_cast<char*>(data_ptr)));
                    }
                }
            },
            {
                CL_PLATFORM_NAME,
                "Platform name",
                [](const std::string& label, void*data_ptr){
                    if(data_ptr){
                        Profiler::add_info(label + ": " + std::string(static_cast<char*>(data_ptr)));
                    }
                }
            },
            {
                CL_PLATFORM_PROFILE,
                "Platform profile",
                [](const std::string& label, void*data_ptr){
                    if(data_ptr){
                        Profiler::add_info(label + ": " + std::string(static_cast<char*>(data_ptr)));
                    }
                }
            },
            {
                CL_PLATFORM_VERSION,
                "Platform version",
                [](const std::string& label, void*data_ptr){
                    if(data_ptr){
                        Profiler::add_info(label + ": " + std::string(static_cast<char*>(data_ptr)));
                    }
                }
            },
        };
        //Printout the platfrom info
        for(auto& platform_info_fetch: platform_info_fetches){
            //Get platform data size and the data => print it out
            clw::ErrorOr<size_t> info_size = platform.info_size(platform_info_fetch.info_id);
            if(info_size.ok()){
                void* platform_info_fetch_data = malloc(info_size.value());
                if(platform.info(platform_info_fetch_data, info_size.value(), platform_info_fetch.info_id) == CL_SUCCESS){
                    platform_info_fetch.data_handler(platform_info_fetch.data_label, platform_info_fetch_data);
                }
                free(platform_info_fetch_data);
            }
        }
    }

    //Prints information about the device
    void print_device_info(clw::Device& device){
        //Device name, hardware version, driver version, OpenCL C version, Parallel compute units
        const std::vector<info_fetch> device_info_fetches = {
            {
                CL_DEVICE_NAME,
                "Device name",
                [](const std::string& label, void*data_ptr){
                    if(data_ptr){
                        Profiler::add_info(label + ": " + std::string(static_cast<char*>(data_ptr)));
                    }
                }
            },
            {
                CL_DEVICE_TYPE,
                "Device type",
                [](const std::string& label, void*data_ptr){
                    if(data_ptr){
                        cl_device_type* type = static_cast<cl_device_type*>(data_ptr);
                        std::string type_bitfield = ""; 
                        std::vector<std::pair<std::string, cl_device_type>> types ={
                            {"DEFAULT", CL_DEVICE_TYPE_DEFAULT},
                            {"CPU", CL_DEVICE_TYPE_CPU},
                            {"GPU", CL_DEVICE_TYPE_GPU},
                            {"ACCELERATOR", CL_DEVICE_TYPE_ACCELERATOR},
                            {"CUSTOM", CL_DEVICE_TYPE_CUSTOM},
                            {"ALL", CL_DEVICE_TYPE_ALL}
                        };
                        //Pull out all the types of the device
                        for(const auto& data_pair: types){
                            if(*type & data_pair.second){
                                type_bitfield += " (" + data_pair.first + ")";
                            }
                        }
                        Profiler::add_info(label + ": " + type_bitfield);
                    }
                }
            },
            {
                CL_DEVICE_VERSION,
                "Device hardware version",
                [](const std::string& label, void*data_ptr){
                    if(data_ptr){
                        Profiler::add_info(label + ": " + std::string(static_cast<char*>(data_ptr)));
                    }
                }
            },
            {
                CL_DRIVER_VERSION,
                "Device driver version",
                [](const std::string& label, void*data_ptr){
                    if(data_ptr){
                        Profiler::add_info(label + ": " + std::string(static_cast<char*>(data_ptr)));
                    }
                }
            },
            {
                CL_DEVICE_OPENCL_C_VERSION,
                "Device openCL_C version",
                [](const std::string& label, void*data_ptr){
                    if(data_ptr){
                        Profiler::add_info(label + ": " + std::string(static_cast<char*>(data_ptr)));
                    }
                }
            },
            {
                CL_DEVICE_MAX_COMPUTE_UNITS,
                "Device Parallel Compute units",
                [](const std::string& label, void*data_ptr){
                    if(data_ptr){
                        Profiler::add_info(label + ": " + std::to_string(*static_cast<cl_uint*>(data_ptr)));
                    }
                }
            },
            {
                CL_DEVICE_LOCAL_MEM_TYPE,
                "Device local memory type",
                [](const std::string& label, void*data_ptr){
                    if(data_ptr){
                        cl_device_local_mem_type* type = static_cast<cl_device_local_mem_type*>(data_ptr);
                        Profiler::add_info(label + ": " + (*type == CL_LOCAL ? "LOCAL" : "GLOBAL"));
                    }
                }
            },
            {
                CL_DEVICE_LOCAL_MEM_SIZE,
                "Device max local memory size",
                [](const std::string& label, void*data_ptr){
                    if(data_ptr){
                        Profiler::add_info(label + ": " + std::to_string(*static_cast<cl_uint*>(data_ptr)));
                    }
                }
            },
            {
                CL_DEVICE_MAX_CLOCK_FREQUENCY,
                "Device max clock frequency",
                [](const std::string& label, void*data_ptr){
                    if(data_ptr){
                        Profiler::add_info(label + ": " + std::to_string(*static_cast<cl_uint*>(data_ptr)));
                    }
                }
            },
            {
                CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                "Device max constant buffer size",
                [](const std::string& label, void*data_ptr){
                    if(data_ptr){
                        Profiler::add_info(label + ": " + std::to_string(*static_cast<cl_uint*>(data_ptr)));
                    }
                }
            },
            {
                CL_DEVICE_MAX_WORK_GROUP_SIZE,
                "Device max work group size",
                [](const std::string& label, void*data_ptr){
                    if(data_ptr){
                        Profiler::add_info(label + ": " + std::to_string(*static_cast<cl_uint*>(data_ptr)));
                    }
                }
            },
            {
                CL_DEVICE_MAX_WORK_ITEM_SIZES,
                "Device max work item sizes",
                [](const std::string& label, void*data_ptr){
                    if(data_ptr){
                        size_t * sizes = static_cast<size_t*>(data_ptr);
                        Profiler::add_info(
                            label + ": " + std::to_string(sizes[0]) + "," +
                            std::to_string(sizes[1]) + "," +
                            std::to_string(sizes[2])
                    );
                    }
                }
            },
        };
        //Printout the platfrom info
        for(auto& device_info_fetch: device_info_fetches){
            //Get device data size and the data => print it out
            clw::ErrorOr<size_t> info_size = device.info_size(device_info_fetch.info_id);
            if(info_size.ok()){
                void* device_info_fetch_data = malloc(info_size.value());
                if(device.info(device_info_fetch_data, info_size.value(), device_info_fetch.info_id) == CL_SUCCESS){
                    device_info_fetch.data_handler(device_info_fetch.data_label, device_info_fetch_data);
                }
                free(device_info_fetch_data);
            }
        }
    }

    //Lists the opencl information of the computer
    void list_info(){
        //Get platforms
        std::vector<clw::Platform> platforms_available;
        get_platforms(platforms_available);
        //Note platform count
        Profiler::add_info("Platform count: " + std::to_string(platforms_available.size()));
        for(auto& platform: platforms_available){
            print_platform_info(platform);
            //Get devices from the platform
            std::vector<clw::Device> platform_devices;
            get_devices(platform, platform_devices);
            //note the count
            Profiler::add_info("Device count: " + std::to_string(platform_devices.size()) + "\n");
            for(auto& device: platform_devices){
                print_device_info(device);
            }
        }
    }

    //Load the opencl kernel from a file
    cl_int initialize(OpenCLRuntime& runtime){
        ScopeTimer scope_timer("add_matrix_initialize");
        const std::string kernel_name = "add_matrix";
        const std::string program_path = "data/kernels/phase_2.cl";
        clw::Device device = prefered_device();
        return runtime.load_file(device, program_path, {kernel_name});
    }

    //Binds the constant arguments of the pipeline
    cl_int bind_pipeline_args(
        OpenCLRuntime& runtime, const int matrix_size
    ){
        //Kernels
        std::shared_ptr<clw::Kernel> addition_kernel = runtime.kernels[0];
        std::vector<std::pair<std::shared_ptr<clw::Kernel>, std::vector<std::pair<int, int>>>> kernel_binds = 
        {
        {
                addition_kernel, std::vector<std::pair<int,int>>{
                    {3, matrix_size}
                }
            },
        };

        cl_int error_code = CL_SUCCESS;
        //Bind the arguments
        for(auto& kernel_arg_pair : kernel_binds){
            for(auto& index_arg_pair : kernel_arg_pair.second){
                if((error_code = kernel_arg_pair.first->bind_arg<int>(index_arg_pair.first, index_arg_pair.second, sizeof(int))) != CL_SUCCESS){
                    return error_code;
                }
            }
        }
        return CL_SUCCESS;
    }

    //Create all the buffers for the pipeline
    cl_int create_buffers(OpenCLRuntime& runtime, std::vector<clw::Buffer>& buffers, const size_t matrix_memory_size){
        //Buffer descriptions
        std::vector<clw::BufferDescription> descriptions = {
            {CL_MEM_READ_ONLY, matrix_memory_size},
            {CL_MEM_READ_ONLY, matrix_memory_size},
            {CL_MEM_WRITE_ONLY, matrix_memory_size}
        };
        //Reserve buffer memory
        buffers.reserve(descriptions.size());
        //Create buffers
        return runtime.context->create_buffers(buffers, descriptions);
    }

    //Binds the given buffers to the kernel as arguments
    cl_int bind_buffers(std::shared_ptr<clw::Kernel> kernel, std::vector<std::pair<int, std::reference_wrapper<clw::Buffer>>> bind_data){
        cl_int error_code = CL_SUCCESS;
        //Bind the buffers to the kernel
        for(auto& bind_pair : bind_data){
            if((error_code = kernel->bind_arg(bind_pair.first, bind_pair.second)) != CL_SUCCESS){
                break;
            }
        }
        return error_code;
    }

    //Queue work into the command queueu based ont he given argumentation
    clw::ErrorOr<std::shared_ptr<clw::Event>> queue_work(
        OpenCLRuntime& runtime, std::shared_ptr<clw::Kernel> kernel, size_t work_dimensions, size_t * global_work_size, size_t * local_size,
        std::vector<std::shared_ptr<clw::Event>> conditions, std::vector<std::pair<int, std::reference_wrapper<clw::Buffer>>> bind_data
    ){
        cl_int error_code = CL_SUCCESS;
        if ((error_code = bind_buffers(kernel, bind_data)) != CL_SUCCESS) {
            return error_code;
        }
        //Start the kernel with buffer_write as wait condition
        return runtime.cc_queue->execute_kernel(kernel, work_dimensions, nullptr, global_work_size, local_size, conditions);
    }

    //Runs the matrix add kernel
    cl_int add_matrix_pipeline(OpenCLRuntime& runtime, float * m1, float * m2, const int matrix_size){
        const size_t matrix_memory_size = matrix_size * sizeof(float);
        float * output = static_cast<float*>(malloc(matrix_memory_size));
        if(output && runtime.kernels.size() > 0){
            std::vector<clw::Buffer> buffers;
            cl_int error_code = 0;

            if(
                (error_code = create_buffers(runtime, buffers, matrix_memory_size)) != CL_SUCCESS ||
                (error_code = bind_pipeline_args(runtime, matrix_size)) != CL_SUCCESS
            ){
                Profiler::add_info(std::string("Initial pipeline setup failed: ") + std::to_string(error_code));
                return error_code;
            }

            std::shared_ptr<clw::Kernel> addition_kernel = runtime.kernels[0];

            //Write matrix data to device
            clw::ErrorOr<std::shared_ptr<clw::Event>> matrix_write1 = runtime.cc_queue->write_buffer(buffers[0], false, 0, matrix_memory_size, m1, {});
            clw::ErrorOr<std::shared_ptr<clw::Event>> matrix_write2 = runtime.cc_queue->write_buffer(buffers[1], false, 0, matrix_memory_size, m2, {});
            if(!matrix_write1.ok()){
                Profiler::add_info(std::string("Matrix write one queueu failed: ") + std::to_string(matrix_write1.error()));
                return matrix_write1.error();
            }
            if(!matrix_write2.ok()){
                Profiler::add_info(std::string("Matrix write two queueu failed: ") + std::to_string(matrix_write2.error()));
                return matrix_write2.error();
            }

            //Queue the addition calculation
            size_t global_work_size[1] = {static_cast<size_t>(matrix_size)};
            clw::ErrorOr<std::shared_ptr<clw::Event>> addition = queue_work(
                runtime, addition_kernel, 1, global_work_size, nullptr,
                 {matrix_write1.value(), matrix_write2.value()}, {{0,buffers[0]}, {1,buffers[1]}, {2, buffers[2]}}
            );
            if(!addition.ok()){
                Profiler::add_info(std::string("Matrix addition queueu failed: ") + std::to_string(addition.error()));
                return addition.error();
            }

            //Read the result
            clw::ErrorOr<std::shared_ptr<clw::Event>> result_read = runtime.cc_queue->read_buffer(buffers[2], false, 0, matrix_memory_size, output, {addition.value()});
            if(!result_read.ok()){
                Profiler::add_info(std::string("Result read queueu failed: ") + std::to_string(result_read.error()));
                return result_read.error();
            }

            //Wait for pipeline to finish
            clw::Event::wait({result_read.value()});
            Profiler::add_info("OpenCL result checksum: " + std::to_string(simple_checksum_float_array(output, matrix_size)));

            //Report timings
            report_cl_timing("Pipeline | matrix one write", matrix_write1.value());
            report_cl_timing("Pipeline | matrix two write", matrix_write2.value());
            report_cl_timing("Pipeline | matrix addition", addition.value());
            report_cl_timing("Pipeline | matrix read", result_read.value());

            free(output);
            return CL_SUCCESS;
        }
        return -1;
    }


}