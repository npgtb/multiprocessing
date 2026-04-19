find_package(OpenCL REQUIRED)

SET(OPENCL_WRAPPER_SRC
    src/platform.cc
    src/device.cc
    src/context.cc
    src/command_queue.cc
    src/program.cc
    src/kernel.cc
    src/image.cc
    src/buffer.cc
    src/event.cc
)

add_library(OPENCL_WRAPPER STATIC ${OPENCL_WRAPPER_SRC})
target_include_directories(OPENCL_WRAPPER
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(OPENCL_WRAPPER PRIVATE OpenCL::OpenCL)