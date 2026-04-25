#include <profiler.h>
#include <opencl_runtime.h>


namespace mp{
    //Init empty
    OpenCLRuntime::OpenCLRuntime():context(nullptr), program(nullptr), kernels(), cc_queue(nullptr){}

    //Try to load from file
    cl_int OpenCLRuntime::load_file(clw::Device& device, const std::string& path, const std::vector<std::string>& kernel_names){
        cl_int error_code = 0;
        std::vector<clw::Device> device_list {device};
        //Create context
        clw::ErrorOr<std::shared_ptr<clw::Context>> context_creation = clw::Device::create_context(device_list);
        if(!context_creation.ok()){
            Profiler::add_info("context creation failed!");
            return context_creation.error();
        }
        //Create commandqueue
        const cl_queue_properties cc_queue_properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0}; //zero terminated
        clw::ErrorOr<std::shared_ptr<clw::CommandQueue>> cc_queue_creation = context_creation.value()->create_cmd_queue(device, cc_queue_properties);
        if(!cc_queue_creation.ok()){
            Profiler::add_info("cc_queue creation failed!");
            return cc_queue_creation.error();
        }
        //Load program from the given path
        clw::ErrorOr<std::shared_ptr<clw::Program>> program_load = context_creation.value()->load_program_file(path);
        if(!program_load.ok()){
            Profiler::add_info("program_load failed from " + path);
            return program_load.error();
        }

        //Try to build the program
        if((error_code = program_load.value()->build_program(device_list)) != CL_SUCCESS){
            Profiler::add_info("program build failed (" + path + ") with code: " + std::to_string(error_code));
            clw::ErrorOr<std::string> build_log = program_load.value()->get_build_log(device_list[0]);
            if(build_log.ok()){
                Profiler::add_info("program build log (" + path + ") is: " + build_log.value());
            }
            return error_code;
        }

        //Create kernels
        for(const auto& kernel_name: kernel_names){
            clw::ErrorOr<std::shared_ptr<clw::Kernel>> kernel_creation = program_load.value()->create_kernel(kernel_name);
            if(!kernel_creation.ok()){
                Profiler::add_info("kernel_creation failed!");
                return kernel_creation.error();
            }
            kernels.push_back(std::move(kernel_creation).value());
        }
        //move the clw values to permanent storage to prevent being collected once they go out of scope
        context = std::move(context_creation).value();
        cc_queue = std::move(cc_queue_creation).value();
        program = std::move(program_load).value();
        return CL_SUCCESS;
    }

}