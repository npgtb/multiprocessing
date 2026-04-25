cmake_minimum_required(VERSION 3.20)
set(THIS multiprocessing)

project(${THIS} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(ENABLE_ASANLEAK)
    message(STATUS "Address sanitizer + leak enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=leak -fno-omit-frame-pointer")
    set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address -fsanitize=leak")
endif()

add_subdirectory(modules)
add_subdirectory(app)

