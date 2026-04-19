#include <SDL3/SDL_timer.h>

#include <drawkit/high_performance_timer.h>

namespace drawkit{

    //Initialize empty timer
    HighPerformanceTimer::HighPerformanceTimer():start_ticks_(0){}

    //Starts the timer
    void HighPerformanceTimer::start(){
        start_ticks_ = SDL_GetPerformanceCounter();
    }

    //Stops the timer, returns time in ms
    double HighPerformanceTimer::stop() const{
        uint64_t stop_ticks = SDL_GetPerformanceCounter();
        return (double)((stop_ticks - start_ticks_ )* 1000) / SDL_GetPerformanceFrequency();
    }

}