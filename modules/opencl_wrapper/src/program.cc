#include <opencl_wrapper/device.h>
#include <opencl_wrapper/program.h>

namespace clw{
    //Initialize empty Program
    Program::Program():context_(nullptr), program_(nullptr){}

    //Initialize Program with id
    Program::Program(cl_program program, std::shared_ptr<Context> context):context_(context), program_(program){}

    //Copy constructors
    Program::Program(Program && other):program_(other.program_), context_(other.context_){
        other.program_ = nullptr;
        other.context_ = nullptr;
    }

    //Release memory
    Program::~Program(){
        release();
    }

    //Assignment operators
    Program& Program::operator=(Program&& other){
        program_ = other.program_;
        context_ = other.context_;
        other.program_ = nullptr;
        other.context_ = nullptr;
        return *this;
    }

    //Frees the memory tied to the cl object
    void Program::release(){
        if(program_){
            clReleaseProgram(program_);
            program_ = nullptr;
            context_ = nullptr;
        }
    }

    //Create a kernel
    ErrorOr<std::shared_ptr<Kernel>> Program::create_kernel(const std::string& name){
        cl_int error_code = 0;
        //https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clCreateKernel.html
        //compiles the source code: program_id, kernel name, error code
        std::shared_ptr<Kernel> kernel = std::make_shared<Kernel>(
            clCreateKernel(program_, name.c_str(), &error_code), shared_from_this()
        );
        if(error_code == CL_SUCCESS){
            return kernel;
        }
        return error_code;
    }

    //Converts CompileOption[] to std::string
    std::string GenerateCompileOtionString(std::vector<clw::Program::CompileOption>& options){
        std::string compile_options = "";
        //Loop trough options appending them to the string
        for(const auto& option : options){
            if(!option.name.empty()){
                compile_options += "-" + option.name;
                if(!option.value.empty()){
                    compile_options += "=" + option.value;
                }
                compile_options += " ";
            }
        }
        if(!compile_options.empty()){
            //Pop last space and null terminate
            compile_options.pop_back();
            compile_options += '\0';
        }
        return compile_options;
    }

    //Tries to build the given program for the given devices
    cl_int Program::build_program(std::vector<Device>& build_devices, std::vector<CompileOption>& options){
        std::vector<cl_device_id> device_ids;
        device_ids.reserve(build_devices.size());
        for(const auto& device: build_devices){
            device_ids.push_back(device.id);
        }
        const char * c_compile_options = nullptr;
        std::string compile_options = GenerateCompileOtionString(options);
        if(!compile_options.empty()){
            c_compile_options = compile_options.c_str();
        }
        //https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clBuildProgram.html
        //compiles the source code: program_id, device_count, devices, build options ,call back func, call back data 
        return clBuildProgram(program_, device_ids.size(), device_ids.data(), c_compile_options, NULL, NULL);
    }

    //Retrieve the build log of the program on the device
    ErrorOr<std::string> Program::get_build_log(clw::Device& device){
        size_t build_log_size = 0;
        cl_int error_code = 0;
        if((error_code = clGetProgramBuildInfo(
                program_, device.id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &build_log_size
        )) == CL_SUCCESS){
            std::string build_log(build_log_size + 1, '\0');
            if((error_code = clGetProgramBuildInfo(
                    program_, device.id, CL_PROGRAM_BUILD_LOG, build_log_size, build_log.data(), nullptr
            )) == CL_SUCCESS){
                return build_log;
            }
            return error_code;
        }
        return error_code;
    }

}