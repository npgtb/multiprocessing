#ifndef OPENCL_WRAPPER_LOCAL_H
#define OPENCL_WRAPPER_LOCAL_H

#include <opencl_wrapper/opencl.h>
#include <opencl_wrapper/error_or.h>

namespace clw{

    struct Local{
        const size_t size;

        //Default constructor
        Local():size(0){}

        //Constructor just takes size
        Local(const size_t size_of_local):size(size_of_local){}
    };

}

#endif