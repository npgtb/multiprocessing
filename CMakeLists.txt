cmake_minimum_required(VERSION 3.20)
set(THIS mp_course)

project(${THIS} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(ENABLE_ASANLEAK)
    message(STATUS "Address sanitizer + leak enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=leak -fno-omit-frame-pointer")
    set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address -fsanitize=leak")
endif()

if(DISABLE_AVX512)
    message(STATUS "Disabling AVX512 instructions")
    add_compile_options(-mno-avx512f)
    add_compile_options(-mno-avx512vl)
    add_compile_options(-mno-avx512bw)
    add_compile_options(-mno-avx512dq)
endif()


add_subdirectory(modules)
add_subdirectory(app)

