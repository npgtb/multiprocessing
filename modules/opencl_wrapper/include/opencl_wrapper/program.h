#ifndef OPENCL_WRAPPER_PROGRAM_H
#define OPENCL_WRAPPER_PROGRAM_H

#include <memory>
#include <vector>
#include <opencl_wrapper/opencl.h>
#include <opencl_wrapper/kernel.h>
#include <opencl_wrapper/error_or.h>

namespace clw{
    class Device;
    class Context;
    class Program : public std::enable_shared_from_this<Program>{
        public:

            //Compiler option struct
            struct CompileOption{
                std::string name;
                std::string value;
            };

            //Initialize empty Program
            Program();

            //Initialize Program with id
            Program(cl_program program, std::shared_ptr<Context> context);

            //Copy constructors
            Program(Program && other);
            Program(const Program&) = delete;

            //Release memory
            ~Program();

            //Assignment operators
            Program& operator=(Program&& other);
            Program& operator=(const Program&) = delete;

            //Frees the memory tied to the cl object
            void release();

            //Tries to build the given program for the given devices
            cl_int build_program(std::vector<Device>& build_devices, std::vector<CompileOption>& options);

            //Create a kernel
            ErrorOr<std::shared_ptr<Kernel>> create_kernel(const std::string& name);

            //Retrieve the build log of the program on the device
            ErrorOr<std::string> get_build_log(clw::Device& device);

        private:
            cl_program program_;
            std::shared_ptr<Context> context_;
    };
}

#endif