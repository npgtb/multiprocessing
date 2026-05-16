#ifndef OPENCL_WRAPPER_EVENT_H
#define OPENCL_WRAPPER_EVENT_H

#include <memory>
#include <vector>
#include <opencl_wrapper/opencl.h>
#include <opencl_wrapper/error_or.h>

namespace clw{

    class CommandQueue;
    class Event{
        public:

            //Initialize empty
            Event();

            //Initialize from cc_queue event
            Event(cl_event& event, std::shared_ptr<CommandQueue> queue_);

            //Copy constructors
            Event(Event && other);
            Event(const Event&) = delete;

            //Release memory
            ~Event();

            //Assignment operators
            Event& operator=(Event&& other);
            Event& operator=(const Event&) = delete;

            //Release the memory
            void release();
            
            //Retrieves the given profiling information
            ErrorOr<cl_ulong> profiling_info(cl_profiling_info info);

            //Waits for the events to finnish executing
            static cl_int wait(std::vector<std::shared_ptr<Event>> events);

        private:
            //friend queue so we can queue things pipeline fashion.
            friend CommandQueue;
            cl_event event_;
            std::shared_ptr<CommandQueue> queue_;
    };
}

#endif