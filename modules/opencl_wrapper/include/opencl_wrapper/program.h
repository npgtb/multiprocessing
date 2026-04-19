#ifndef OPENCL_WRAPPER_PROGRAM_H
#define OPENCL_WRAPPER_PROGRAM_H

#include <memory>
#include <opencl_wrapper/opencl.h>
#include <opencl_wrapper/kernel.h>
#include <opencl_wrapper/error_or.h>

namespace clw{
    class Context;
    class Program : public std::enable_shared_from_this<Program>{
        public:
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

            //Create a kernel
            ErrorOr<std::shared_ptr<Kernel>> create_kernel(const std::string& name);

        private:
            cl_program program_;
            std::shared_ptr<Context> context_;
    };
}

#endif