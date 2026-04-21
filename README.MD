# MULTIPROCESSING
Experimentation with opencl multiprocessing
## DEPENDENCIES
### ```modules\drawkit``` 
This module depends on the [SDL3](https://github.com/libsdl-org/SDL) and [SDL3_Image](https://github.com/libsdl-org/SDL_image) projects.
### ```modules\opencl_wrapper```
This module depends on the opencl project, specifically the 3.0 version.
## Building
The ```CMakePresets.json``` provides a release preset for building with ```Ninja``` build system.
```phase_3_vectorized.cc``` utilizes ```__builtin_cpu_supports``` which is supported by gcc and clang.
```
cmake --preset release
cmake --build --preset release
```
## Running
The built application expects to be given two image paths as arguments
```
./build/release/app/multiprocessing path/to/img0.png path/to/img1.png
```
## Output
The application outputs a detailed description of its activities to the console and to a log file: ```profiler_output.log```. Raw timings are also
outputted to a .csv file: ```sample_data.csv```. ZNCC result images are output as a .png file to the folder.
## Code map
### ```modules\drawkit```
A pruned SDL based library for handling loading images formats.
### ```modules\opencl_wrapper```
A RAII - style C++ wrapping library for the Opencl C-API. I know there is
official bindings for C++, but i figured this was a better way to learn.
### ```app/*/cpu_workloads```
Contains the code for the work to be done on the cpu in phases 2, 3 and 4.
- ```phase_2.cc```: matrix addition on cpu
- ```phase_3.cc```: single threaded zncc
- ```phase_3_vectorized.cc```: single threaded zncc, utilizing the SSE and AVX extensions for vectorization
- ```phase_4.cc```: multithreaded zncc via threadpool
- ```phase_4_vectorized.cc```: multithreaded zncc via threadpool, utilizing the SSE and AVX extensions for vectorization
### ```app/*/opencl_workloads```
Contains the code for work to be done via opencl in phase 1,2,5,6 and 7.
- ```phase_1.cc```: opencl hello world
- ```phase_2.cc```: opencl information pulling and matrix addition.
- ```phase_5.cc```: ZNCC implementation using global memory.
- ```phase_6_image2d_t.cc```: ZNCC implementation using the image2d opencl object.
- ```phase_6_vectorized.cc```: ZNCC implementation doing local tiling and vectorized loads.
- ```phase_6_tiled.cc```: ZNCC implementation doing local tiling and focusing on register pressure.
### ```app/*/*```
Supporting functionality for executing the different workloads:
- ```combined_workloads.cc```: Collects different workloads under single functions, for easy running.
- ```helpers.cc```: Different helper functions dealing with opencl platforms and devices, reporting timings from opencl events and checksumming float arrays.
- ```image.cc```: Data structure for holding image pixel data in memory.
- ```opencl_runtime.cc```: Collects the opencl program initialization tasks under the same roof: context creation, command queueu creation, program loading and building, creating the kernels from the program.
- ```thread_pool.cc```: simple thread pool implementation.
- ```profiler.cc```: Static profiling/logging functionality provider.
- ```scope_timer.cc```: Logs the 'scopes' execution time on desctruction.

