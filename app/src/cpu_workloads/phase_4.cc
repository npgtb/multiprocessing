#include <math.h>
#include <algorithm>
#include <scope_timer.h>
#include <cpu_workloads/phase_4.h>

namespace mp_course::cpu_workloads::phase_4{

    //Generic function for queueing linear work into the threadpool
    template <typename Function, typename... Arguments>
    void queue_linear_work(
        ThreadPool& thread_pool, int work_size,
        Function&& work, Arguments&&... arguments
    ){
        if(work_size > 0){
            //Calculate work size
            const int thread_count = std::min(work_size, thread_pool.pool_size());
            const int work_chunk_size = std::floor(work_size /  thread_count);
            //Queue work
            for(int i = 0; i < (thread_count-1); ++i){
                const int chunk_start = (work_chunk_size * i);
                const int chunk_end = (work_chunk_size * (i+1));
                thread_pool.queue_work(
                    std::forward<Function>(work),
                    std::forward<Arguments>(arguments)...,
                    chunk_start, chunk_end
                );
            }
            //Queue the remainder of the work into last thread
            const int remainder_start = work_chunk_size * (thread_count - 1);
            thread_pool.queue_work(
            std::forward<Function>(work),
                std::forward<Arguments>(arguments)...,
                remainder_start, work_size
            );
        }
    }

    //Chunk processor for resize_image
    void resize_image_work_chunk(
        uint32_t* original, uint32_t* downsized, const int new_width, const int new_height, const int original_width, const int factor,
        const int work_row_start, const int work_row_end
    ){
        //Process the given rows
        for(int j = work_row_start; j < work_row_end; ++j){
            for(int i = 0; i < new_width; ++i){
                downsized[j * new_width + i] = original[(j*factor) * original_width + (i * factor)];
            }
        }
    }

    //Resize the image by factor. Takes the nth row and column approach. Expects a RGBA format image.
    bool resize_image(Image& image, const int factor, ThreadPool& thread_pool){
        if(image.format == ImageFormat::RGBA){
            mp_course::ScopeTimer exec_timer("mp_course::zncc_c_multi_thread::resize_image");
            //Calculate new dimensions and allocate memory
            const int new_width = image.w / factor;
            const int new_height = image.h / factor;
            uint32_t * original_image = static_cast<uint32_t*>(image.pixels);
            uint32_t * downsized_image = static_cast<uint32_t*>(malloc(new_width * new_height * sizeof(uint32_t)));

            if(downsized_image){
                //Threading strat => Split image into chunks of X rows, each thread owns those rows in the new image
                //where X = (image_rows / pool size) (+ remainder added to the last workload)
                queue_linear_work(
                    thread_pool, new_height, resize_image_work_chunk,
                    original_image, downsized_image, new_width, new_height, image.w, factor
                );
                //Wait for work to finish
                thread_pool.wait_for_work();
                original_image = nullptr;
                image.free_memory();
                image.w = new_width;
                image.h = new_height;
                image.pixels = static_cast<void*>(downsized_image);
                return true;
            }
        }
        return false;
    }

    //Chunk processor for grayscale_image
    void grayscale_image_work_chunk(uint32_t* rgba_pixels, uint8_t* grayscaled_pixels, const int chunk_start, const int chunk_end){
        //Pixel modifiers
        constexpr int red_shift = 24, green_shift = 16, blue_shift = 8;
        constexpr float red_modifier = 0.2126, green_modifier = 0.7152, blue_modifier = 0.0722;
        constexpr uint8_t mask = 0xFF;
        //Process the chunk
        for(int i = chunk_start; i < chunk_end; ++i){
            //Apply the grayscale formula to the image pixels
            grayscaled_pixels[i] = static_cast<uint8_t>(
                std::round(
                    ((rgba_pixels[i] >> red_shift) & mask) * red_modifier + 
                    ((rgba_pixels[i] >> green_shift) & mask) * green_modifier +
                    ((rgba_pixels[i] >> blue_shift) & mask) * blue_modifier
                )
            );
        }
    }

    //Grayscales the image. Expects a RGBA format image and turns image to GRAY format.
    bool grayscale_image(Image& image, ThreadPool& thread_pool){
        if(image.format == ImageFormat::RGBA){
            mp_course::ScopeTimer exec_timer("mp_course::zncc_c_multi_thread::grayscale_image");
            //allocate memory
            const int memory_size = image.w * image.h;
            uint8_t * grayscaled_pixels = static_cast<uint8_t*>(malloc(memory_size * sizeof(uint8_t)));
            if(grayscaled_pixels){
                uint32_t * rgba_pixels = static_cast<uint32_t*>(image.pixels);
                //Threading strat => Split image into chunks of X pixels
                //where X = (total_pixels / pool size) (+ remainder added to the last workload)
                queue_linear_work(
                    thread_pool, memory_size, grayscale_image_work_chunk,
                    rgba_pixels, grayscaled_pixels
                );
                //Wait for work to finish
                thread_pool.wait_for_work();
                rgba_pixels = nullptr;
                image.free_memory();
                image.pixels = static_cast<void*>(grayscaled_pixels);
                image.format = ImageFormat::GRAY;
                return true;
            }
        }
        return false;
    }

    //Calculates mean from the window in the given image. Expects grayscale image
    float calculate_window_mean(int x, int y, int radius, Image& image){
        uint8_t* pixels = static_cast<uint8_t*>(image.pixels);
        int window_size = (radius * 2 + 1);
        int count = window_size * window_size;
        float sum = 0;
        for(int yr = -radius; yr <= radius; ++yr){
            int cy = image.clamp_y(y + yr) * image.w ;
            for(int xr = -radius; xr <= radius; ++xr){
                //Edge handling => nearest valid pixel
                int cx = image.clamp_x(x + xr);
                sum += pixels[cy + cx];
            }
        }
        return sum / count;
    }

    //Calculates the ZNCC trough upper and lower sums. Expects grayscaled images
    float calculate_zncc(int x, int rx, int y, int radius, float lmean, float rmean, Image& left, Image& right){
        uint8_t* lpixels = static_cast<uint8_t*>(left.pixels);
        uint8_t* rpixels = static_cast<uint8_t*>(right.pixels);
        float upper = 0.f;
        float lower_l = 0.f, lower_r = 0.f;
        for(int yr = -radius; yr <= radius; ++yr){
            int cy = left.clamp_y(y + yr) * left.w;
            for(int xr = -radius; xr <= radius; ++xr){
                //Edge handling => nearest valid pixel
                int lcx = left.clamp_x(x + xr);
                int rcx = right.clamp_x(rx + xr);
                //Calculate difference
                float l_diff = lpixels[cy + lcx] - lmean;
                float r_diff = rpixels[cy + rcx] - rmean;
                //Adjust upper and lower sums
                upper += l_diff * r_diff;
                lower_l += (l_diff * l_diff);
                lower_r += (r_diff * r_diff);
            }
        }

        float divider = sqrt(lower_l) * sqrt(lower_r);
        if(divider != 0){
            return upper / divider;
        }
        return 0.f;
    }

    //Chunk processor for calculate_disparity_map
    void calculate_disparity_map_work_chunk(
        int window_radius, int min_disparity, int max_disparity, const bool left_to_right, Image& left, Image& right, Image& map, const int chunk_start, const int chunk_end
    ){
        uint8_t* disparity_map_pixels = static_cast<uint8_t*>(map.pixels);
        int disparity_direction = 1;
        if(left_to_right){
            disparity_direction = -1;
        }
        //Calculate ZNCC
        for(int y = chunk_start; y < chunk_end; ++y){
            for(int x = 0; x < left.w; ++x){
                float max_zncc = -1.f; 
                uint8_t zncc_max_disparity = 0;
                //Calculate Left mean
                float left_mean = calculate_window_mean(x, y, window_radius, left);
                //Right window x coordinate: x - disparity
                for(int d = min_disparity; d <= max_disparity; ++d){
                    int rx = x + (d * disparity_direction);
                    //Right mean into, zncc calculation
                    float right_mean = calculate_window_mean(rx, y, window_radius, right);
                    float zncc = calculate_zncc(x, rx, y, window_radius, left_mean, right_mean, left, right);
                    //See if we have new highscore for zncc
                    if(zncc > max_zncc){
                        max_zncc = zncc;
                        zncc_max_disparity = d;
                    }
                }
                disparity_map_pixels[y * map.w + x] = zncc_max_disparity;
            }
        }
    }

    //Calculates the disparity using the ZNCC algo. Calculates disparity shift from left image to right image, storing values in map image.
    bool calculate_disparity_map(int window_radius, int min_disparity, int max_disparity, const bool left_to_right, Image& left, Image& right, Image& map, ThreadPool& thread_pool, std::string scope_tag){
        if(left.w == right.w && left.h == right.h && left.format == ImageFormat::GRAY && right.format == ImageFormat::GRAY && window_radius > 0){
            mp_course::ScopeTimer exec_timer("mp_course::zncc_c_multi_thread::calculate_disparity_map_" + scope_tag);
            //Allocate disparity map
            map.free_memory();
            map.w = left.w; map.h = left.h;
            map.format = ImageFormat::GRAY;
            map.pixels = malloc(map.w * map.h);
            if(map.pixels){
                uint8_t* disparity_map_pixels = static_cast<uint8_t*>(map.pixels);
                //Threading strat => Split the work into Chunks of X rows.
                queue_linear_work(
                    thread_pool, left.h, calculate_disparity_map_work_chunk,
                    window_radius, min_disparity, max_disparity, left_to_right, std::ref(left), std::ref(right), std::ref(map)
                );
                //Wait for work to finish
                thread_pool.wait_for_work();
                return true;
            }
        }
        return false;
    }

    //Maps the disparity value scale to grayscale
    uint8_t grayscale_disparity(
        const uint8_t min_disparity, const uint8_t max_disparity, const uint8_t value
    ){
        if(value <= min_disparity){
            return 0;
        }
        return 255.f * (value - min_disparity) / (max_disparity - min_disparity); 
    }

    //Calculate middle value using histogram
    uint8_t calculate_window_non_zero_middle(int x, int y, int radius, Image& image){
        //Assume grayscale
        uint8_t* pixels = static_cast<uint8_t*>(image.pixels);
        int count = 0;
        std::array<int, 256> histogram = {};
        //Count the values with in the buckets
        for(int yr = -radius; yr <= radius; ++yr){
            int cy = image.clamp_y(y + yr) * image.w;
            for(int xr = -radius; xr <= radius; ++xr){
                //Edge handling => nearest valid pixel
                int cx = image.clamp_x(x + xr);
                uint8_t pixel_value = pixels[cy + cx];
                if(pixel_value > 0){
                    histogram[pixel_value]++;
                    count++;
                }
            }
        }
        //Do we have values in the histogram buckets?
        if(count > 0){
            //Middle value is reached when weve seen X pixels
            int middle_count = count >> 1;
            int * histogram_value = &histogram[1];
            int sum = 0;
            for(int i = 1; i < 256; ++i){
                sum += *histogram_value++;
                if(sum > middle_count){
                    return i;
                }
            }
        }
        return 0;
    }

    //Chunk processor for calculate_disparity_map
    void cross_check_occulsion_disparity_maps_work_chunk(
        int threshold_value, int window_radius, int min_disparity, int max_disparity, Image& left_disparity, Image& right_disparity, Image& pp_disparity,
        const int chunk_start, const int chunk_end
    ){
        uint8_t * left_pixels = static_cast<uint8_t*>(left_disparity.pixels);
        uint8_t * right_pixels = static_cast<uint8_t*>(right_disparity.pixels);
        uint8_t * pp_pixels = static_cast<uint8_t*>(pp_disparity.pixels);

        //Cross check and occuld
        for(int y = chunk_start; y < chunk_end; ++y){
            for(int x = 0; x < left_disparity.w; ++x){ 
                uint8_t disparity_value_l = left_pixels[y * left_disparity.w + x];
                //In the right map, we move to the left disparity_value_l mutch
                uint8_t disparity_value_r = 0;
                if(x - disparity_value_l >= 0){
                    disparity_value_r = right_pixels[y * left_disparity.w + x - disparity_value_l];
                }

                uint8_t final_value = disparity_value_l;
                //Cross-check and occuld
                if(abs(disparity_value_l - disparity_value_r) > threshold_value || final_value == 0){
                    final_value = calculate_window_non_zero_middle(x, y, window_radius, left_disparity);
                }
                pp_pixels[y * left_disparity.w + x] = grayscale_disparity(min_disparity, max_disparity, final_value);
            }
        }
    }

    //Cross-checks the two disparity maps against each other
    bool cross_check_occulsion_disparity_maps(int threshold_value, int window_radius, int min_disparity, int max_disparity, Image& left_disparity, Image& right_disparity, Image& pp_disparity, ThreadPool& thread_pool){
        if(
            left_disparity.format == ImageFormat::GRAY && right_disparity.format == ImageFormat::GRAY &&
            left_disparity.w == right_disparity.w && left_disparity.h == right_disparity.h
        ){
            mp_course::ScopeTimer exec_timer("mp_course::zncc_c_multi_thread::cross_check_occulsion_disparity_maps");
            //reserve space for the post processed map
            pp_disparity.free_memory();
            pp_disparity.pixels = malloc(left_disparity.w * left_disparity.h);
            if(pp_disparity.pixels){
                pp_disparity.format = ImageFormat::GRAY;
                pp_disparity.w = left_disparity.w;
                pp_disparity.h = left_disparity.h;
                //Threading strat => Split the work into Chunks of X rows.
                queue_linear_work(
                    thread_pool, left_disparity.h, cross_check_occulsion_disparity_maps_work_chunk,
                    threshold_value, window_radius, min_disparity, max_disparity, std::ref(left_disparity), std::ref(right_disparity), std::ref(pp_disparity)
                );
                //Wait for work to finish
                thread_pool.wait_for_work();
                return true;
            }
        }
        return false;
    }

}