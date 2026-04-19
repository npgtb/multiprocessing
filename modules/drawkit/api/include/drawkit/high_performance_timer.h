#ifndef DRAWKIT_API_HIGH_PERFORMANCE_TIMER_H
#define DRAWKIT_API_HIGH_PERFORMANCE_TIMER_H

#include <cstdint>

namespace drawkit{
    class HighPerformanceTimer{
        public:
            //Initialize empty timer
            HighPerformanceTimer();
        
            //Starts the timer
            void start();

            //Stops the timer, returns time in ms
            double stop() const;

        private:
            uint64_t start_ticks_;
    };

}

#endif