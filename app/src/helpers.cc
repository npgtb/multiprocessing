#include <helpers.h>
#include <profiler.h>

namespace mp_course{

    //Simple checksum for float array comparision
    double simple_checksum_float_array(float * arr, const int size){
        double checksum = 0.0;
        for(int i = 0; i < size; ++i){
            checksum += (arr[i] * i);
        }
        return checksum;
    }

    //Reports the timing information of the given event to the profiler
    void report_cl_timing(const std::string& label, std::shared_ptr<clw::Event> event){
        clw::ErrorOr<cl_ulong> queued = event->profiling_info(CL_PROFILING_COMMAND_QUEUED);
        clw::ErrorOr<cl_ulong> submit = event->profiling_info(CL_PROFILING_COMMAND_SUBMIT);
        clw::ErrorOr<cl_ulong> start = event->profiling_info(CL_PROFILING_COMMAND_START);
        clw::ErrorOr<cl_ulong> end = event->profiling_info(CL_PROFILING_COMMAND_END);

        if(!queued.ok()){
            Profiler::add_info( label + ", CL_PROFILING_COMMAND_QUEUED Error code: " + std::to_string(queued.error()));
            return;
        }
        if(!submit.ok()){
            Profiler::add_info( label + ", CL_PROFILING_COMMAND_SUBMIT Error code: " + std::to_string(submit.error()));
            return;
        }
        if(!start.ok()){
            Profiler::add_info( label + ", CL_PROFILING_COMMAND_START Error code: " + std::to_string(start.error()));
            return;
        }
        if(!end.ok()){
            Profiler::add_info( label + ", CL_PROFILING_COMMAND_END Error code: " + std::to_string(end.error()));
            return;
        }
        
        double queued_to_submit = (submit.value() - queued.value()) * 1e-6;
        double submit_to_start = (start.value() - submit.value()) * 1e-6;
        double execution = (end.value() - start.value()) * 1e-6;
        double total = (end.value() - queued.value()) * 1e-6;

        //CL timings are in nanoseconds => convert to milliseconds
        Profiler::add_additional_timing(label + " [QUEUE to SUBMIT]", queued_to_submit, false);
        Profiler::add_additional_timing(label + " [SUBMIT to START]", submit_to_start, false);
        Profiler::add_additional_timing(label + " [QUEUE to END]", total, false);
        Profiler::add_timing(label + " [EXECUTION]", execution, false);
    }


    //Gets available platforms
    void get_platforms(std::vector<clw::Platform>& platforms){
        clw::ErrorOr<std::vector<clw::Platform>> available_platforms = clw::Platform::available();
        if(available_platforms.ok()){
            platforms = std::move(available_platforms).value();
        }
    }

    //Gets devices from the given platform
    void get_devices(clw::Platform platform, std::vector<clw::Device>& devices){
        clw::ErrorOr<std::vector<clw::Device>> available_devices = platform.devices();
        if(available_devices.ok()){
            devices = std::move(available_devices).value();
        }
    }

    //Find a computing device that is prefered, preference order: GPU, INTEGRATED, CPU
    clw::Device prefered_device(){
        //Get platforms
        std::vector<clw::Platform> platforms;
        get_platforms(platforms);
        //Find prefered device
        clw::Device integrated_gpu, cpu;
        for(auto& platform : platforms){
            //Get devices
            std::vector<clw::Device> devices;
            get_devices(platform, devices);
            for(auto& device: devices){
                //Query type
                cl_device_type device_type;
                if(device.info(&device_type, sizeof(cl_device_type), CL_DEVICE_TYPE) == CL_SUCCESS){
                    if(device_type & CL_DEVICE_TYPE_GPU){
                        //Find out if the device shares memory with the host
                        cl_bool shared_memory = false;
                        if(device.info(&shared_memory, sizeof(cl_bool), CL_DEVICE_HOST_UNIFIED_MEMORY) == CL_SUCCESS){
                            if(!shared_memory){
                                //Dedicated GPU
                                return device;
                            }
                            integrated_gpu = device;
                        }
                        else if(device_type & CL_DEVICE_TYPE_CPU){
                            cpu = device;
                        }
                    }
                }

            }
        }
        //INTEGRATED GPU => CPU
        if(integrated_gpu.id != 0){
            return integrated_gpu;
        }
        return cpu;
    }

}