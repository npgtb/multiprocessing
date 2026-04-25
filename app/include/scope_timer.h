#include <string>
#include <drawkit/high_performance_timer.h>

namespace mp{
    class ScopeTimer{
        public:
            //Start timing
            ScopeTimer(const std::string& label);

            //Stop timing, report result
            ~ScopeTimer();

        private:
            std::string label_;
            drawkit::HighPerformanceTimer timer_;
    };
}