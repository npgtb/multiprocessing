#include <fstream>
#include <algorithm>
#include <profiler.h>

namespace mp{

    //Initialize empty profiler
    Profiler::Profiler()
    :total_host_(0), total_device_(0), segment_host_(0), segment_device_(0), scope_host_(0), scope_device_(0), total_indentation_(0), segment_indentation_(2), scope_indentation_(4){}

    //Singelton instance
    Profiler& Profiler::instance(){
        static Profiler profiler;
        return profiler;
    }

    //Appends a new timing to the profiler ouput
    void Profiler::add_timing(const std::string& label, double timing, bool host_timing){
        Profiler& profiler = instance();
        Segment& current_segment = profiler.find_sample(profiler.current_segment_, profiler.timing_samples_);
        Scope& current_scope = profiler.find_sample(profiler.current_scope_, current_segment.samples);
        profiler.find_sample(label, current_scope.samples).push_back({.is_host = host_timing, .is_additional = false, .timing = timing});
    }

    //Appends a new timing to the profiler ouput
    void Profiler::add_additional_timing(const std::string& label, double timing, bool host_timing){
        Profiler& profiler = instance();
        Segment& current_segment = profiler.find_sample(profiler.current_segment_, profiler.timing_samples_);
        Scope& current_scope = profiler.find_sample(profiler.current_scope_, current_segment.samples);
        profiler.find_sample(label, current_scope.samples).push_back({.is_host = host_timing, .is_additional = true, .timing = timing});
    }

    //Appends the provided information to the profiling output
    void Profiler::add_info(const std::string& info){
        Profiler& profiler = instance();
        std::string new_info_string = std::string(profiler.scope_indentation_, ' ') + "[INFO] " + info + "\n";
        if(std::find(profiler.info_strings_.begin(), profiler.info_strings_.end(), new_info_string) == profiler.info_strings_.end()){
            profiler.info_strings_.push_back(new_info_string);
        }
    }

    //Add a output stream to the logger reporting
    void Profiler::add_output(std::shared_ptr<std::ostream> stream){
        instance().output_.push_back(stream);
    }

    //Starts new segment
    void Profiler::segment_start(const std::string& segment_name){
        Profiler& profiler = instance();
        profiler.current_segment_ = segment_name;
    }


    void Profiler::scope_start(const std::string& label){
        Profiler& profiler = instance();
        profiler.current_scope_ = label;
    }

    //Returns the results of the ran profilers to the streams
    void Profiler::output(){
        Profiler& profiler = instance();
        //Output info strings
        for(const auto& info_string : profiler.info_strings_){
            profiler.profilings_ << info_string;
        }
        //Output the timings
        profiler.total_host_ = 0, profiler.total_device_ = 0;
        for(auto& segment_pair : profiler.timing_samples_){
            profiler.output_segment(segment_pair.first, segment_pair.second);
        }
        profiler.profilings_ << "\n[PROFILER END] Total host: " << profiler.total_host_ << " ms, total device: " << profiler.total_device_ << " ms\n";
        std::string contents = profiler.profilings_.str();
        for(auto stream : profiler.output_){
            *stream << contents;
            stream->flush();
        }
    }

    //Outputs the samples as a csv file
    void Profiler::output_csv(const std::string& path){
        Profiler& profiler = instance();
        std::fstream output(path, std::ios::out | std::ios::app);
        if(output.is_open()){
            for(auto& segment_pair : profiler.timing_samples_){
                for(auto& scope_pair : segment_pair.second.samples){
                    for(auto& timing_pair : scope_pair.second.samples){
                        for(const auto& timing : timing_pair.second){
                            output << segment_pair.first << "," 
                                   << scope_pair.first << ","
                                   << timing_pair.first << ","
                                   << timing.is_host << ","
                                   << timing.is_additional << ","
                                   << timing.timing << "\n";
                        }
                    }
                }
            }
        }
    }

    //Output the data of the segment
    void Profiler::output_segment(const std::string& segment_name, const Segment& segment){
        Profiler& profiler = instance();
        profiler.segment_host_ = 0, profiler.segment_device_ = 0;
        profiler.profilings_ << "\n" << std::string(8, '-') << "[SEGMENT START] " 
                    << segment_name << std::string(8, '-') << "\n";
        for(auto& scope_pair : segment.samples){
            profiler.output_scope(scope_pair.first, scope_pair.second);
        }
        profiler.profilings_ << std::string(8, '-') << "[SEGMENT END] " << "total host: " << profiler.segment_host_ 
                            << " ms, total device: " << profiler.segment_device_ << " ms" << std::string(8, '-') << "\n";
        profiler.total_host_ += profiler.segment_host_;
        profiler.total_device_ += profiler.segment_device_;
    }

    //Output the data of the scope
    void Profiler::output_scope(const std::string& scope_name, const Scope& segment){
        Profiler& profiler = instance();
        profiler.scope_host_ = 0, profiler.scope_device_ = 0;
        profiler.profilings_ << std::string(4, '-')  << "[SCOPE START] " << scope_name << std::string(4, '-') << "\n";
        for(auto& timing_pair : segment.samples){
            profiler.output_timing(timing_pair.first, timing_pair.second);
        }
        profiler.profilings_ << std::string(4, '-') << "[SCOPE END] " << "total host: " << profiler.scope_host_ 
                            << " ms, total device: " << profiler.scope_device_ << " ms" << std::string(4, '-') << "\n";
        profiler.segment_host_ += profiler.scope_host_;
        profiler.segment_device_ += profiler.scope_device_;
    }

    //Output the data of the timing
    void Profiler::output_timing(const std::string label, const std::vector<Timing>& timings){
        Profiler& profiler = instance();
        double host_average = calculate_average(timings, true);
        double host_middle = calculate_middle(timings, true);

        double device_average = calculate_average(timings, false);
        double device_middle = calculate_middle(timings, false);

        if(!timings[0].is_additional){
            if(host_average > 0.0){
                profiler.scope_host_ += host_average;
                profiler.profilings_ << std::string(profiler.scope_indentation_, ' ') << "[HOST] " << label << " Avg " << host_average << "ms, "
                                     << " Middle " << host_middle << "ms\n";
            }
            if(device_average > 0.0){
                profiler.scope_device_ += device_average;
                profiler.profilings_ << std::string(profiler.scope_indentation_, ' ') << "[DEVICE] " << label << " Avg " << device_average << "ms, "
                                     << "Middle " << device_middle << "ms\n";
            }
        }
        else{
            if(host_average > 0.0){
                profiler.profilings_ << std::string(profiler.scope_indentation_, ' ') << "[HOST ADDITIONAL] " << label << " Avg " << host_average << "ms, "
                                     << " Middle " << host_middle << "ms\n";
            }
            if(device_average > 0.0){
                profiler.profilings_ << std::string(profiler.scope_indentation_, ' ') << "[DEVICE ADDITIONAL] " << label << " Avg " << device_average << "ms, "
                                     << "Middle " << device_middle << "ms\n";
            }
        }
    }

    //Calculate average of the timings
    double Profiler::calculate_average(const std::vector<Timing>& timings, bool is_host){
        double sum = 0.0;
        int count = 0;
        for(const auto& timing : timings){
            if(timing.is_host == is_host){
                sum += timing.timing;
                ++count;
            }
        }
        if(count == 0){
            return 0.0;
        }
        return sum/count;
    }

    //Calculate the middle value of the timings
    double Profiler::calculate_middle(const std::vector<Timing>& timings, bool is_host){
        std::vector<double> valid_timings;
        for(const auto& timing : timings){
            if(timing.is_host == is_host){
                valid_timings.push_back(timing.timing);
            }
        }
        if(!valid_timings.empty()){
            const int timings_count = valid_timings.size();
            std::nth_element(valid_timings.begin(), valid_timings.begin() + timings_count / 2, valid_timings.end());
            if(timings_count % 2 == 1){
                return valid_timings[timings_count / 2];
            }
            return (valid_timings[timings_count / 2 - 1] + valid_timings[timings_count / 2]) / 2;
        }
        return 0.0;
    }

}