#include <opencl_wrapper/event.h>

namespace clw{
    //Initialize empty
    Event::Event():event_(nullptr), queue_(nullptr){}

    //Initialize from cc_queue event
    Event::Event(cl_event& event, std::shared_ptr<CommandQueue> queue_):event_(event), queue_(queue_){}

    //Copy constructors
    Event::Event(Event && other):event_(other.event_), queue_(other.queue_){
        other.event_ = nullptr;
        other.queue_ = nullptr;
    }

    //Release memory
    Event::~Event(){
        release();
    }

    //Assignment operators
    Event& Event::operator=(Event&& other){
        event_ = other.event_;
        queue_ = other.queue_;
        other.event_ = nullptr;
        other.queue_ = nullptr;
        return *this;
    }

    //Release the memory
    void Event::release(){
        if(event_){
            clReleaseEvent(event_);
            event_ = nullptr;
            queue_ = nullptr;
        }
    }

    //Retrieves the given profiling information
    ErrorOr<cl_ulong> Event::profiling_info(cl_profiling_info info){
        cl_ulong timing_info = 0;
        cl_int error_code = clGetEventProfilingInfo(event_, info, sizeof(timing_info), &timing_info, nullptr);
        if(error_code == CL_SUCCESS){
            return timing_info;
        }
        return error_code;
    }

    //Waits for the events to finnish executing
    cl_int Event::wait(std::vector<std::shared_ptr<Event>> events){
        //Pull out the cl_event ids
        std::vector<cl_event> cl_events;
        cl_events.reserve(events.size());
        for(auto& event :events){
            cl_events.push_back(event->event_);
        }
        //Wait for the events
        return clWaitForEvents(cl_events.size(), cl_events.data());
    }

}