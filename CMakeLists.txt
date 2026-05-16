cmake_minimum_required(VERSION 3.20)
set(THIS multiprocessing)

project(${THIS} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(modules)
add_subdirectory(app)

