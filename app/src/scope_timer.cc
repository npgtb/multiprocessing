#include <profiler.h>
#include <scope_timer.h>

namespace mp{
    //Start timing
    ScopeTimer::ScopeTimer(const std::string& label):label_(label){
        Profiler::scope_start(label);
        timer_.start();
    }

    //Stop timing, report result
    ScopeTimer::~ScopeTimer(){
        Profiler::add_timing(label_, timer_.stop(), true);
    }
}