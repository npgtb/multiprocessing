#include <vector>
#include <memory>
#include <sstream>

namespace mp{
    enum class TimingDevice{
        CPU, GPU
    };

    class Profiler{
        public:

            //Initialize empty profiler
            Profiler();

            //Singelton instance
            static Profiler& instance();

            //Appends a new timing to the profiler ouput
            static void add_timing(const std::string& label, double timing, bool host_timing);

            //Appends a new timing to the profiler ouput, not counted in the total
            static void add_additional_timing(const std::string& label, double timing, bool host_timing);

            //Appends the provided information to the profiling output
            static void add_info(const std::string& info);

            //Add a output stream to the logger reporting
            static void add_output(std::shared_ptr<std::ostream> stream);

            //Starts new segment
            static void segment_start(const std::string& segment_name);

            //Start new scope
            static void scope_start(const std::string& label);

            //Returns the results of the ran profilers to the streams
            static void output();

            //Outputs the samples as a csv file
            static void output_csv(const std::string& path);

        private:

            double total_host_, total_device_; 
            double segment_host_, segment_device_;
            double scope_host_, scope_device_;

            int total_indentation_, segment_indentation_, scope_indentation_;

            std::stringstream profilings_;
            std::vector<std::shared_ptr<std::ostream>> output_;

            std::string current_scope_;
            std::string current_segment_;
            std::vector<std::string> info_strings_;

            //Holds info about the timings
            struct Timing{
                bool is_host;
                bool is_additional;
                double timing;
            };

            //Holds samples for the scope
            struct Scope{
                std::vector<std::pair<std::string, std::vector<Timing>>> samples;
            };

            //Holds the samples for the segment
            struct Segment{
                std::vector<std::pair<std::string, Scope>> samples;
            };

            //Holds the segments
            std::vector<std::pair<std::string, Segment>> timing_samples_;

            //Find a sample using the label of it
            template <typename T>
            T& find_sample(const std::string& label, std::vector<std::pair<std::string, T>>& samples){
                for(auto& pairing : samples){
                    if(pairing.first == label){
                        return pairing.second;
                    }
                }
                samples.emplace_back(label, T());
                return samples.back().second;
            }

            //Output the data of the segment
            void output_segment(const std::string& segment_name, const Segment& segment);

            //Output the data of the scope
            void output_scope(const std::string& scope_name, const Scope& segment);

            //Output the data of the timing
            void output_timing(const std::string label, const std::vector<Timing>& timings);

            //Calculate average of the timings
            double calculate_average(const std::vector<Timing>& timings, bool is_host);

            //Calculate the middle value of the timings
            double calculate_middle(const std::vector<Timing>& timings, bool is_host);
    };
}
