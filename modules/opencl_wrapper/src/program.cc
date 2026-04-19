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

}