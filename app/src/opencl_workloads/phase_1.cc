#include <helpers.h>
#include <profiler.h>
#include <scope_timer.h>
#include <opencl_runtime.h>
#include <opencl_workloads/phase_1.h>

namespace mp::gpu_workloads::phase_1{

    //Load the opencl kernel from a file
    cl_int hello_world_load(OpenCLRuntime& cl_single){
        ScopeTimer scope_timer("hello_world_load");
        const std::string kernel_name = "hello_world";
        const std::string program_path = "data/kernels/phase_1.cl";
        clw::Device device = prefered_device();
        std::vector<clw::Program::CompileOption> compile_options{};
        return cl_single.load_file(device, program_path, {kernel_name}, compile_options);
    }

    //Bind the arguments to the opencl kernel
    cl_int hello_world_setup(clw::Buffer& buffer, OpenCLRuntime& cl_single, const size_t output_buffer_size){
        ScopeTimer scope_timer("hello_world_setup");
        const int output_buffer_arg_index = 0;
        if(cl_single.kernels.size() > 0){
            const auto& kernel = cl_single.kernels[0];
            //Create a opencl output buffer
            clw::ErrorOr<clw::Buffer> buffer_creation = cl_single.context->create_buffer(CL_MEM_WRITE_ONLY, output_buffer_size);
            if(!buffer_creation.ok()){
                return buffer_creation.error();    
            }
            buffer = std::move(buffer_creation).value();
            //Bind buffer and queue the work
            return kernel->bind_arg(output_buffer_arg_index, buffer);
        }
        return CL_INVALID_VALUE;
    }

    //Run the opencl kernel
    cl_int hello_world_run(const size_t output_buffer_size, char * output_bfr, clw::Buffer& buffer, OpenCLRuntime& cl_single){
        ScopeTimer scope_timer("hello_world_run");
        constexpr size_t global_work_dimensions = 1;
        size_t global_work_size[global_work_dimensions] = {1};
        if(cl_single.kernels.size() > 0){
            const auto& kernel = cl_single.kernels[0];
            clw::ErrorOr<std::shared_ptr<clw::Event>> kernel_exec_queue = cl_single.cc_queue->execute_kernel(kernel, global_work_dimensions, nullptr, global_work_size, nullptr, {});
            //Execute the kernel
            if(kernel_exec_queue.ok()){
                //Read output
                clw::ErrorOr<std::shared_ptr<clw::Event>> buffer_read = cl_single.cc_queue->read_buffer(buffer, false, 0, output_buffer_size, output_bfr, {});
                if(buffer_read.ok()){
                    //Wait for exec and read
                    clw::Event::wait({kernel_exec_queue.value(), buffer_read.value()});
                    report_cl_timing("Kernel execution", kernel_exec_queue.value());
                    report_cl_timing("Buffer read", buffer_read.value());
                    return CL_SUCCESS;
                }
                return buffer_read.error();
            }
            return kernel_exec_queue.error();
        }
        return CL_INVALID_VALUE;
    }

    //Runs the hello world kernel
    void hello_world(){
        constexpr size_t output_buffer_size = 13;
        char output_buffer[output_buffer_size];
        cl_int error_code = 0;
        OpenCLRuntime cl_single;
        if((error_code = hello_world_load(cl_single)) == CL_SUCCESS){
            clw::Buffer buffer;
            if((error_code = hello_world_setup(buffer, cl_single, output_buffer_size)) == CL_SUCCESS){
                if((error_code = hello_world_run(output_buffer_size, output_buffer, buffer, cl_single)) == CL_SUCCESS){
                    Profiler::add_info(std::string("hello_world Kernel Output: ") + output_buffer);
                }
            }
        }
        //Output any errors
        if(error_code != CL_SUCCESS){
            Profiler::add_info(std::string("hello_world Opencl error code: ") + std::to_string(error_code));
        }
    }
}