#include <math.h>
#include <algorithm>
#include <immintrin.h>
#include <scope_timer.h>
#include <cpu_workloads/phase_4.h>

namespace mp_course::cpu_workloads::phase_4_vectorized{

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
#if defined(__SSE4_1__)
    //Grayscale function utilizing the SSE 128 wide registers
    void grayscale_image_sse_128_work_chunk(uint32_t* rgba_pixels, uint8_t* grayscaled_pixels, const int chunk_start, const int chunk_end){
        constexpr int red_shift = 24, green_shift = 16, blue_shift = 8;
        constexpr int right_shift = 10;
        constexpr uint8_t mask = 0xFF;
        constexpr int red_modifier = 218, green_modifier = 732, blue_modifier = 74;
        //Broadcast 32 bit mask value to the 128 bit value
        const __m128i mask_128 = _mm_set1_epi32(mask);
        //Broadcast 32 bit modifier values to the 128 bit variable 
        const __m128i red_modifier_128 = _mm_set1_epi32(red_modifier);
        const __m128i green_modifier_128 = _mm_set1_epi32(green_modifier);
        const __m128i blue_modifier_128 = _mm_set1_epi32(blue_modifier);

        int i = chunk_start;
        //Process 4 pixel at time (128 bits)
        for(; i + 4 < chunk_end; i += 4){
            //Load 128 bits from the rgba values
            __m128i pixels = _mm_loadu_si128((__m128i*)(rgba_pixels + i));
            //Pull out the individual color channels from the pixels by shifting each 32 bit value and mask anding the 128 bit value
            __m128i red_channel = _mm_and_si128(_mm_srli_epi32(pixels, red_shift), mask_128);
            __m128i green_channel = _mm_and_si128(_mm_srli_epi32(pixels, green_shift), mask_128);
            __m128i blue_channel = _mm_and_si128(_mm_srli_epi32(pixels, blue_shift), mask_128);
            //Multiply the packed 32 bit integers
            red_channel = _mm_mullo_epi32(red_channel, red_modifier_128);
            green_channel = _mm_mullo_epi32(green_channel, green_modifier_128);
            blue_channel = _mm_mullo_epi32(blue_channel, blue_modifier_128);
            //Sum the packed 32 bit values
            __m128i channel_sum = _mm_add_epi32(_mm_add_epi32(red_channel, green_channel), blue_channel);
            //Shift the packed 32 bit values down to 8 bit values
            channel_sum = _mm_srli_epi32(channel_sum, right_shift);
            //Pack 32 bits down to 8 bit values
            __m128i packed16 = _mm_packs_epi32(channel_sum, _mm_setzero_si128());
            __m128i packed8  = _mm_packus_epi16(packed16, packed16);
            //Store the lower 32 bit value
            *(uint32_t*)(grayscaled_pixels + i) = _mm_cvtsi128_si32(packed8);
        }
        //Process the remainder 
        for (; i < chunk_end; ++i) {
            uint32_t pixel = rgba_pixels[i];
            uint8_t red_chanel = (pixel >> red_shift) & mask;
            uint8_t green_channel = (pixel >> green_shift) & mask;
            uint8_t blue_channel = (pixel >> blue_shift) & mask;
            grayscaled_pixels[i] = (red_modifier * red_chanel + green_modifier * green_channel + blue_modifier * blue_channel) >> 10;
        }
    }
#endif

#if defined(__AVX512F__)
    //Grayscale function utilizing the AVX 512 wide registers
    void grayscale_image_avx_512_work_chunk(uint32_t* rgba_pixels, uint8_t* grayscaled_pixels, const int chunk_start, const int chunk_end){

        //Set modifiers and allocate memory
        constexpr int red_shift = 24, green_shift = 16, blue_shift = 8;
        constexpr int right_shift = 10;
        constexpr uint8_t mask = 0xFF;
        constexpr int red_modifier = 218, green_modifier = 732, blue_modifier = 74;
        //Broadcast 32 bit mask value to the 512 bit value
        const __m512i mask_512 = _mm512_set1_epi32(mask);
        //Broadcast 32 bit modifier values to the 128 bit variable 
        const __m512i red_modifier_512 = _mm512_set1_epi32(red_modifier);
        const __m512i green_modifier_512 = _mm512_set1_epi32(green_modifier);
        const __m512i blue_modifier_512 = _mm512_set1_epi32(blue_modifier);

        int i = chunk_start;
        //Process 4 pixel at time (128 bits)
        for(; i + 16 < chunk_end; i += 16){
            //Load 512 bits from the rgba values
            __m512i pixels = _mm512_loadu_si512((__m512i*)(rgba_pixels + i));
            //Pull out the individual color channels from the pixels by shifting each 32 bit value and mask anding the 512 bit value
            __m512i red_channel = _mm512_and_si512(_mm512_srli_epi32(pixels, red_shift), mask_512);
            __m512i green_channel = _mm512_and_si512(_mm512_srli_epi32(pixels, green_shift), mask_512);
            __m512i blue_channel = _mm512_and_si512(_mm512_srli_epi32(pixels, blue_shift), mask_512);
            //Multiply the packed 32 bit integers
            red_channel = _mm512_mullo_epi32(red_channel, red_modifier_512);
            green_channel = _mm512_mullo_epi32(green_channel, green_modifier_512);
            blue_channel = _mm512_mullo_epi32(blue_channel, blue_modifier_512);
            //Sum the packed 32 bit values
            __m512i channel_sum = _mm512_add_epi32(_mm512_add_epi32(red_channel, green_channel), blue_channel);
            //Shift the packed 32 bit values down to 8 bit values
            channel_sum = _mm512_srli_epi32(channel_sum, right_shift);
            //Pack 32 bits down to 8 bit values for storing
            //Split the 512 into high and low 256
            __m256i low_32_values = _mm512_castsi512_si256(channel_sum);
            __m256i high_32_values  = _mm512_extracti64x4_epi64(channel_sum, 1);
            //Split the 256's into 128s
            __m128i low_128_1 = _mm256_castsi256_si128(low_32_values);
            __m128i low_128_2 = _mm256_extracti128_si256(low_32_values, 1);
            __m128i high_128_1 = _mm256_castsi256_si128(high_32_values);
            __m128i high_128_2 = _mm256_extracti128_si256(high_32_values, 1);
            //Pack the 32 bit values into 16 bit values
            __m128i low_packed_16 = _mm_packs_epi32(low_128_1, low_128_2);
            __m128i high_packed_16 = _mm_packs_epi32(high_128_1, high_128_2);
            //Pack the 16 bit values into 8 bit values
            __m128i packed_8 = _mm_packus_epi16(low_packed_16, high_packed_16);
            //Store the grayscale values
            _mm_storeu_si128((__m128i*)(grayscaled_pixels + i), packed_8);
        }
        //Process the remainder 
        for (; i < chunk_end; ++i) {
            uint32_t pixel = rgba_pixels[i];
            uint8_t red_chanel = (pixel >> red_shift) & mask;
            uint8_t green_channel = (pixel >> green_shift) & mask;
            uint8_t blue_channel = (pixel >> blue_shift) & mask;

            grayscaled_pixels[i] = (red_modifier * red_chanel + green_modifier * green_channel + blue_modifier * blue_channel) >> 10;
        }
    }
#endif

    //Grayscales the image. Expects a RGBA format image and turns image to GRAY format.
    bool grayscale_image(Image& image, ThreadPool& thread_pool){
        if(image.format == ImageFormat::RGBA){
            mp_course::ScopeTimer exec_timer("mp_course::zncc_c_multi_thread::grayscale_image");
            //allocate memory
            const int memory_size = image.w * image.h;
            uint8_t * grayscaled_pixels = static_cast<uint8_t*>(malloc(memory_size * sizeof(uint8_t)));
            if(grayscaled_pixels){
                uint32_t * rgba_pixels = static_cast<uint32_t*>(image.pixels);
                #if defined(__AVX512F__)
                queue_linear_work(
                    thread_pool, memory_size, grayscale_image_avx_512_work_chunk,
                    rgba_pixels, grayscaled_pixels
                );
                #elif defined(__SSE4_1__)
                queue_linear_work(
                    thread_pool, memory_size, grayscale_image_sse_128_work_chunk,
                    rgba_pixels, grayscaled_pixels
                );
                #else
                    return false;
                #endif
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

#if defined(__SSE4_1__)
    //Calculates mean from the window in the given image. Expects grayscale image
    float calculate_window_mean_sse_128(int x, int y, int radius, Image& image){
        uint8_t* pixels = static_cast<uint8_t*>(image.pixels);
        int window_size = ((radius *2) + 1);
        int pixel_count = window_size * window_size;
        float sum = 0;
        constexpr int load_size = 8;
        
        __m128i zero = _mm_setzero_si128();
        for(int yr = -radius; yr <= radius; ++yr){
            int cy = image.clamp_y(y + yr) * image.w;
            int i = 0; 
            int xr = -radius;
            //Can we load 8 at time, and do we need to clamp?
            for(; 
                  i + load_size < window_size &&
                  i + load_size < image.w && 
                  xr + load_size <= radius &&
                  x + xr >= 0; 
                  i += load_size, xr += load_size
            ){
                int load_index = cy + (x + xr);
                //load first 8 grayscale values as int64
                __m128i pixel_values = _mm_cvtsi64_si128(*(uint64_t*)(pixels + load_index));
                //Sum both lanes to the finalsum
                sum += _mm_cvtsi128_si32(_mm_sad_epu8(pixel_values, zero));
            }
            //Handle remainder
            for(; xr <= radius; ++xr){
                int cx = image.clamp_x(x + xr);
                sum += pixels[cy + cx];
            }
        }
        return sum / pixel_count;
        return 0.f;
    }

    //Calculates the ZNCC trough upper and lower sums. Expects grayscaled images
    float calculate_zncc_sse_128(int x, int y, int radius, int disparity, float lmean, float rmean, Image& left, Image& right){
        uint8_t* lpixels = static_cast<uint8_t*>(left.pixels);
        uint8_t* rpixels = static_cast<uint8_t*>(right.pixels);
        float upper = 0.f;
        float lower_l = 0.f, lower_r = 0.f;
        int window_size = ((radius *2) + 1);
        constexpr int load_size = 8;
        const int rx = x - disparity; 
        __m128i zero = _mm_setzero_si128();
        __m128 left_mean_128 = _mm_set1_ps(lmean);
        __m128 right_mean_128 = _mm_set1_ps(rmean);
        for(int yr = -radius; yr <= radius; ++yr){
            int cy = left.clamp_y(y + yr) * left.w;
            int i = 0; 
            int xr = -radius;
            //Can we load 8 at time, and do we need to clamp?
            for(; 
                  i + load_size < window_size &&
                  i + load_size < left.w && 
                  xr + load_size <= radius &&
                  x + xr >= 0 &&
                  rx + xr >= 0; 
                  i += load_size, xr += load_size
            ){
                int lcx = x + xr;
                int rcx = rx + xr;
                //load first 8 grayscale values as int64
                __m128i left_pixel_values = _mm_cvtsi64_si128(*(uint64_t*)(lpixels + (cy + lcx)));
                __m128i right_pixel_values = _mm_cvtsi64_si128(*(uint64_t*)(rpixels + (cy + rcx)));
                //Unpack low 8 bits as 16 bit wide
                __m128i left_values_low = _mm_unpacklo_epi8(left_pixel_values, zero);
                __m128i right_values_low = _mm_unpacklo_epi8(right_pixel_values, zero);
                //Convert 16 bit (32bit) packed values to floating point values
                __m128 float_left_low = _mm_cvtepi32_ps(_mm_unpacklo_epi16(left_values_low, zero)); 
                __m128 float_left_high = _mm_cvtepi32_ps(_mm_unpackhi_epi16(left_values_low, zero));
                __m128 float_right_low = _mm_cvtepi32_ps(_mm_unpacklo_epi16(right_values_low, zero));
                __m128 float_right_high = _mm_cvtepi32_ps(_mm_unpackhi_epi16(right_values_low, zero));
                //Subtract means
                float_left_low = _mm_sub_ps(float_left_low, left_mean_128);
                float_left_high = _mm_sub_ps(float_left_high, left_mean_128);
                float_right_low = _mm_sub_ps(float_right_low, right_mean_128);
                float_right_high = _mm_sub_ps(float_right_high, right_mean_128);
                //Do the multiplications
                __m128 upper_low = _mm_mul_ps(float_left_low, float_right_low);
                __m128 upper_high = _mm_mul_ps(float_left_high, float_right_high);
                __m128 lower_left_low = _mm_mul_ps(float_left_low, float_left_low);
                __m128 lower_left_high = _mm_mul_ps(float_left_high, float_left_high);
                __m128 lower_right_low = _mm_mul_ps(float_right_low, float_right_low);
                __m128 lower_right_high = _mm_mul_ps(float_right_high, float_right_high);
                //Add to the zncc sums
                //Add 32bit floats in lower and higher together
                __m128 sum_upper = _mm_add_ps(upper_low, upper_high);
                //Move 2 high floats into the lower position and add them to the lower position floats
                sum_upper = _mm_add_ps(sum_upper, _mm_movehl_ps(sum_upper, sum_upper));
                //Shuffle high of low to the low position and add low positions together
                sum_upper = _mm_add_ss(sum_upper, _mm_shuffle_ps(sum_upper, sum_upper, 1));
                ///Extract lowest float
                upper += _mm_cvtss_f32(sum_upper);

                __m128 sum_lower_left = _mm_add_ps(lower_left_low, lower_left_high);
                sum_lower_left = _mm_add_ps(sum_lower_left, _mm_movehl_ps(sum_lower_left, sum_lower_left));
                sum_lower_left = _mm_add_ss(sum_lower_left, _mm_shuffle_ps(sum_lower_left, sum_lower_left, 1));
                lower_l += _mm_cvtss_f32(sum_lower_left);

                __m128 sum_lower_right = _mm_add_ps(lower_right_low, lower_right_high);
                sum_lower_right = _mm_add_ps(sum_lower_right, _mm_movehl_ps(sum_lower_right, sum_lower_right));
                sum_lower_right = _mm_add_ss(sum_lower_right, _mm_shuffle_ps(sum_lower_right, sum_lower_right, 1));
                lower_r += _mm_cvtss_f32(sum_lower_right);
            }
            //Handle remainder
            for(; xr <= radius; ++xr){
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
#endif
    //Chunk processor for calculate_disparity_map
    void calculate_disparity_map_work_chunk(
        int window_radius, int min_disparity, int max_disparity, Image& left, Image& right, Image& map, const int chunk_start, const int chunk_end
    ){
        #if defined(__SSE4_1__)
        uint8_t* disparity_map_pixels = static_cast<uint8_t*>(map.pixels);
        //Calculate ZNCC
        for(int y = chunk_start; y < chunk_end; ++y){
            for(int x = 0; x < left.w; ++x){
                float max_zncc = -1.f; 
                uint8_t zncc_max_disparity = 0;
                //Calculate Left mean
                float left_mean = calculate_window_mean_sse_128(x, y, window_radius, left);
                //Right window x coordinate: x - disparity
                for(int d = min_disparity, rx = x - min_disparity; d <= max_disparity; ++d, rx--){
                    //Right mean into, zncc calculation
                    float right_mean = calculate_window_mean_sse_128(rx, y, window_radius, right);
                    float zncc = calculate_zncc_sse_128(x, y, window_radius, d, left_mean, right_mean, left, right);
                    //See if we have new highscore for zncc
                    if(zncc > max_zncc){
                        max_zncc = zncc;
                        zncc_max_disparity = d;
                    }
                }
                disparity_map_pixels[y * map.w + x] = zncc_max_disparity;
            }
        }
        #endif
    }

    //Calculates the disparity using the ZNCC algo. Calculates disparity shift from left image to right image, storing values in map image.
    bool calculate_disparity_map(int window_radius, int min_disparity, int max_disparity, Image& left, Image& right, Image& map, ThreadPool& thread_pool, std::string scope_tag){
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
                    window_radius, min_disparity, max_disparity, std::ref(left), std::ref(right), std::ref(map)
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

    //Calculate medium value using histogram
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