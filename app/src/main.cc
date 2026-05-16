#include <iostream>
#include <fstream>
#include <profiler.h>
#include <drawkit/init.h>
#include <combined_workloads.h>
#include <cpu_workloads/cpu_workloads.h>
#include <opencl_workloads/opencl_workloads.h>

//Run work
void run_work(const std::string& left, const std::string& right, mp::ThreadPool& thread_pool){
    constexpr int resize_factor = 4, window_radius = 4, min_disparity = 0, max_disparity = 65, cross_check_threshold = 8, sample_count = 1;
    mp::gpu_workloads::run_hello_world_workload();
    mp::gpu_workloads::run_list_info_workload();
    mp::run_combined_add_matrix_workload(sample_count);
    mp::run_combined_zncc_workloads(
    left, right, resize_factor, window_radius, min_disparity, 
        max_disparity, cross_check_threshold, sample_count, thread_pool
    );
}

//Initializes the thread pool
bool init_thread_pool(mp::ThreadPool& thread_pool){
    //Use the number of logical cores in the cpu as number of threads in the pool
    const int thread_count = std::thread::hardware_concurrency();
    mp::Profiler::add_info("Thread pool initialized to " + std::to_string(thread_count) + " threads");
    return thread_pool.initialize(thread_count);
}

//Initializes the profiler output streams
bool init_profiler(){
    //Report profiler results to a log file and console
    std::shared_ptr<std::fstream> log_file = std::make_shared<std::fstream>("profiler_output.log", std::ios::out | std::ios::app);
    std::shared_ptr<std::ostream> console_stream(&std::cout, [](std::ostream*){}); // Cant close std::cout
    if(log_file->is_open()){
        mp::Profiler::add_output(log_file);
        mp::Profiler::add_output(console_stream);
        return true;
    }
    return false;
}

int main(int argc, char** args){
    if(argc == 3){
        if(drawkit::init() && init_profiler()){
            mp::ThreadPool thread_pool;
            if(init_thread_pool(thread_pool)){
                //Assume image path at args[1] && args[2]
                run_work(args[1], args[2], thread_pool);
                mp::Profiler::output();
                mp::Profiler::output_csv("sample_data.csv");
            }
        }
        drawkit::shutdown();
    }
    return 0;
}
