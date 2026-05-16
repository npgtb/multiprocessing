add_executable(${THIS} 
    src/main.cc
    src/scope_timer.cc
    src/profiler.cc
    src/image.cc
    src/opencl_runtime.cc
    src/combined_workloads.cc
    src/helpers.cc
    src/thread_pool.cc

    src/cpu_workloads/cpu_workloads.cc
    src/cpu_workloads/phase_2.cc
    src/cpu_workloads/phase_3.cc
    src/cpu_workloads/phase_3_vectorized.cc
    src/cpu_workloads/phase_4.cc
    src/cpu_workloads/phase_4_vectorized.cc
    src/cpu_workloads/phase_7.cc

    src/opencl_workloads/opencl_workloads.cc
    src/opencl_workloads/phase_1.cc
    src/opencl_workloads/phase_2.cc
    src/opencl_workloads/phase_5.cc
    src/opencl_workloads/phase_6_tiled.cc
    src/opencl_workloads/phase_6_vectorized.cc
    src/opencl_workloads/phase_6_image2d_t.cc
    src/opencl_workloads/phase_7_integral.cc
    src/opencl_workloads/phase_7_integral_aps.cc
    src/opencl_workloads/phase_7_integral_aps_cl.cc
)

if(ENABLE_ASANLEAK)
    message(STATUS "${THIS} Address sanitizer + leak enabled")
    target_compile_options(${THIS} PRIVATE
        -g
        -fsanitize=address
        -fsanitize=leak
        -fno-omit-frame-pointer
    )
    target_link_options(${THIS} PRIVATE
        -fsanitize=address
        -fsanitize=leak
    )
endif()

target_compile_options(${THIS} PRIVATE -march=native)
target_include_directories(${THIS}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(${THIS} PRIVATE DRAWKIT OPENCL_WRAPPER)

#Copy data to program path
add_custom_command(TARGET ${THIS} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/data
        $<TARGET_FILE_DIR:${THIS}>/data
)