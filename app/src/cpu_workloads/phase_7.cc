#include <math.h>
#include <immintrin.h>
#include <scope_timer.h>
#include <cpu_workloads/phase_7.h>

namespace mp::cpu_workloads::phase_7{

    //Chunk processor for resize_image
    void resize_image_work_chunk(
        uint32_t* original, uint32_t* downsized, 
        const int new_width, const int new_height, const int original_width, 
        const int factor, const int chunk_start, const int chunk_end
    ){
        //Process the given rows
        for(int j = chunk_start; j < chunk_end; ++j){
            for(int i = 0; i < new_width; ++i){
                downsized[j * new_width + i] = original[(j*factor) * original_width + (i * factor)];
            }
        }
    }

    //Resize the image by factor. Takes the nth row and column approach. Expects a RGBA format image.
    bool resize_image(Image& input, Image& output, const int factor, ThreadPool& thread_pool){
        if(input.format == ImageFormat::RGBA){
            mp::ScopeTimer exec_timer("mp::zncc_c_multi_thread::resize_image");
            //Calculate new dimensions and allocate memory
            const int new_width = input.w / factor;
            const int new_height = input.h / factor;
            output.init<uint32_t>(new_width, new_height, ImageFormat::RGBA);
            uint32_t * original_image = input.data<uint32_t>();
            uint32_t * downsized_image = output.data<uint32_t>();
            queue_linear_work(
                thread_pool, new_height, resize_image_work_chunk,
                original_image, downsized_image, new_width, new_height, input.w, factor
            );
            //Wait for work to finish
            thread_pool.wait_for_work();
            return true;     
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
            _mm_storeu_si32(grayscaled_pixels + i, packed8);
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
            mp::ScopeTimer exec_timer("mp::zncc_c_multi_thread::grayscale_image");
            //allocate memory
            const int memory_size = image.w * image.h;
            std::vector<uint8_t> gs_data(memory_size);
            uint32_t * rgba_pixels = image.data<uint32_t>();
            uint8_t * gs_pixels = gs_data.data();
    #if defined(__AVX512F__)
            queue_linear_work(
                thread_pool, memory_size, grayscale_image_avx_512_work_chunk,
                rgba_pixels, gs_pixels
            );
    #elif defined(__SSE4_1__)
            queue_linear_work(
                thread_pool, memory_size, grayscale_image_sse_128_work_chunk,
                rgba_pixels, gs_pixels
            );
    #else
            return false;
    #endif
            //Wait for work to finish
            thread_pool.wait_for_work();
            image.set(std::move(gs_data), image.w, image.h, ImageFormat::GRAY);
            return true;
        }
        return false;
    }

    //First pass of the integral image process: ROW
    void calculate_integral_map_row_work_chunk(
        Image& img, Image& map, uint8_t * pixels, uint64_t * sm_data, uint64_t* sq_data,
        const int max_d, const int radius, const bool right,
        const int chunk_start, const int chunk_end
    ){
        //Y-axis pad the halo
        const int y_offset = chunk_start - radius;
        //X-axis pad halo + disparity
        int x_start = right ? (-radius - max_d) : -radius;
        for(int img_y = y_offset, map_y = chunk_start; map_y < chunk_end; ++img_y, ++map_y){
            uint64_t row_sum = 0, row_sqr = 0;
            const int map_row = map_y * map.w;
            //For image clamp y => Replication
            const int img_row = img.clamp_y(img_y) * img.w; 
            for(int img_x = x_start, map_x = 0; map_x < map.w; ++img_x, ++map_x){
                //For image clamp x => Replication
                const int img_i = img_row + img.clamp_x(img_x);
                const int map_i = map_row + map_x;
                uint8_t value = pixels[img_i];
                //Get pixel value
                row_sum += value;
                row_sqr += (value * value);
                //Store running values for the pixel
                sm_data[map_i] = row_sum;
                sq_data[map_i] = row_sqr;
            }
        }
    }

    //Second pass of the integral image process: COLUMNS
    void calculate_integral_map_column_work_chunk(
        Image& img, Image& map, uint64_t * sm_data, uint64_t* sq_data,
        const int radius, const int chunk_start, const int chunk_end
    ){
        for(int map_x = chunk_start; map_x < chunk_end; ++map_x){
            uint64_t column_sum = 0, column_sqr = 0;
            for(int map_y = 0; map_y < map.h; ++map_y){
                const int map_i = (map_y * map.w) + map_x;
                //Get column values
                column_sum += sm_data[map_i];
                column_sqr += sq_data[map_i];
                //Store running values for the pixel
                sm_data[map_i] = column_sum;
                sq_data[map_i] = column_sqr;
            }
        }
    }

    //Precalculate the integral maps for the image
    bool calculate_integral_map(
        Image& image, Image& sum_map, Image& square_map, 
        const int radius, const int max_d, const bool right,
        ThreadPool& pool
    ){
        if(image.format == ImageFormat::GRAY){
            mp::ScopeTimer exec_timer("mp::zncc_c_multi_thread::calculate_integral_map");
            //Allocate memory for the integral maps, account for halo
            const int window_diameter = (radius << 1);
            const int y_pad = radius;
            const int x_pad = (right ? radius + max_d : radius);
            const int map_width = (image.w + window_diameter + max_d);
            const int map_height = (image.h + window_diameter);
            sum_map.init<uint64_t>(map_width, map_height, x_pad, y_pad, ImageFormat::INTEGRAL);
            square_map.init<uint64_t>(map_width, map_height, x_pad, y_pad, ImageFormat::INTEGRAL);
            uint64_t * sm_data = sum_map.data<uint64_t>();
            uint64_t * sq_data = square_map.data<uint64_t>();
            uint8_t * pixels = image.data<uint8_t>();
            //Queue row pass
            queue_linear_work(
                pool, map_height, calculate_integral_map_row_work_chunk,
                std::ref(image), std::ref(sum_map), pixels, sm_data, sq_data, max_d, radius, right
            );
            //Sync for the 2nd pass
            pool.wait_for_work();
            //Queue column pass
            queue_linear_work(
                pool, map_width, calculate_integral_map_column_work_chunk,
                    std::ref(image), std::ref(sum_map), sm_data, sq_data, radius
            );
            pool.wait_for_work();
            //Store data in the maps
            return true;
        }
        return false;
    }

    //Returns the map integral of the area as described
    uint64_t integral_map_sum(
        uint64_t* int_data, const int int_map_w,
        const int center_x, const int center_y, const int radius
    ){
        //Utilizing the formula described in
        //https://en.wikipedia.org/wiki/Summed-area_table
        const int x = std::max(center_x - radius-1, 0);
        const int y = std::max(center_y - radius-1, 0);
        const int x1 = center_x + radius;
        const int y1 = center_y + radius;

        uint64_t D = int_data[(y1 * int_map_w) + x1];
        uint64_t C = int_data[(y1 * int_map_w) + x];
        uint64_t B = int_data[(y * int_map_w) + x1];
        uint64_t A = int_data[(y * int_map_w) + x]; 
        return D + A - B - C; 
    }

    //Returns the mean of the given area
    float integral_mean(
        uint64_t* int_data, const int int_map_w,
        const int x, const int y, const int radius, const int win_pix_count
    ){
        return ((float) integral_map_sum(int_data, int_map_w, x, y, radius)) / win_pix_count;
    }


#if defined(__SSE4_1__)
    //Calculates the ZNCC trough upper and lower sums. Expects grayscaled images
    float calculate_zncc_sse_128(
        Image& d_map, Image& lsq, Image& rsq, 
        uint8_t* l_pixels, uint8_t* r_pixels,
        uint64_t* lsq_data, uint64_t* rsq_data,
        int x, int rx, int y, int radius, 
        float lmean, float rmean, float left_var, 
        int win_pix_count
    ){
        float upper = 0.f;
        __m128 upper_vec = _mm_setzero_ps();
        int window_size = ((radius *2) + 1);
        constexpr int load_size = 8;
        __m128i zero = _mm_setzero_si128();
        for(int yr = -radius; yr <= radius; ++yr){
            int cy = d_map.clamp_y(y + yr) * d_map.w;
            int xr = -radius;
            //Can we load 8 at time, and do we need to clamp?
            for(; 
                x + load_size < d_map.w && 
                rx + load_size < d_map.w &&
                xr + load_size <= radius &&
                x + xr >= 0 &&
                rx + xr >= 0; 
                xr += load_size
            ){
                int lcx = x + xr;
                int rcx = rx + xr;
                //load first 8 grayscale values as int64
                __m128i left_pixel_values = _mm_cvtsi64_si128(*(uint64_t*)(l_pixels + (cy + lcx)));
                __m128i right_pixel_values = _mm_cvtsi64_si128(*(uint64_t*)(r_pixels + (cy + rcx)));
                //Unpack low 8 bits as 16 bit wide
                __m128i left_values_low = _mm_unpacklo_epi8(left_pixel_values, zero);
                __m128i right_values_low = _mm_unpacklo_epi8(right_pixel_values, zero);
                //Convert 16 bit (32bit) packed values to floating point values
                __m128 float_left_low = _mm_cvtepi32_ps(_mm_unpacklo_epi16(left_values_low, zero)); 
                __m128 float_left_high = _mm_cvtepi32_ps(_mm_unpackhi_epi16(left_values_low, zero));
                __m128 float_right_low = _mm_cvtepi32_ps(_mm_unpacklo_epi16(right_values_low, zero));
                __m128 float_right_high = _mm_cvtepi32_ps(_mm_unpackhi_epi16(right_values_low, zero));
                //Do the multiplications
                __m128 upper_low = _mm_mul_ps(float_left_low, float_right_low);
                __m128 upper_high = _mm_mul_ps(float_left_high, float_right_high);
                //Accumilate into upper vec
                upper_vec = _mm_add_ps(upper_vec, upper_low);
                upper_vec = _mm_add_ps(upper_vec, upper_high);
            }
            //Handle remainder
            for(; xr <= radius; ++xr){
                int lcx = d_map.clamp_x(x + xr);
                int rcx = d_map.clamp_x(rx + xr);
                //Adjust upper and lower sums
                upper += l_pixels[cy + lcx] * r_pixels[cy + rcx];
            }

        }
        //Reduce SIMD vector to add to the upper sum
        __m128 sum_upper = upper_vec;
        //Move 2 high floats into the lower position and add them to the lower position floats
        sum_upper = _mm_add_ps(sum_upper, _mm_movehl_ps(sum_upper, sum_upper));
        //Shuffle high of low to the low position and add low positions together
        sum_upper = _mm_add_ss(sum_upper, _mm_shuffle_ps(sum_upper, sum_upper, 1));
        ///Extract lowest float
        upper += _mm_cvtss_f32(sum_upper);

        upper -= win_pix_count * lmean * rmean;
        float right_var = integral_mean(rsq_data, rsq.w, rsq.pad_x(rx), rsq.pad_y(y), radius, win_pix_count) - (rmean * rmean);
        float divider = win_pix_count * sqrt(left_var * right_var);
        if(divider > 0.0){
            return upper / divider;
        }
        return 0.f;
    }

    //Chunk processor for calculate_disparity_map
    void calculate_disparity_map_work_chunk(
        int radius, int min_d, int max_d, const int d_dir, const int win_pix_count,
        Image& left, Image& right, Image& d_map, Image& lsm, Image& rsm, Image& lsq, Image& rsq,
        const int chunk_start, const int chunk_end
    ){
        uint8_t* d_map_pixels = d_map.data<uint8_t>();
        uint8_t* l_pixels = left.data<uint8_t>();
        uint8_t* r_pixels = right.data<uint8_t>();
        uint64_t* lsm_pixels = lsm.data<uint64_t>();
        uint64_t* rsm_pixels = rsm.data<uint64_t>();
        uint64_t* lsq_pixels = lsq.data<uint64_t>();
        uint64_t* rsq_pixels = rsq.data<uint64_t>();
        for(int y = chunk_start; y < chunk_end; ++y){
            for(int x = 0; x < d_map.w; ++x){
                float max_zncc = -1.f; 
                uint8_t zncc_max_d = 0;
                //Calculate Left mean and variance
                float l_mean = integral_mean(lsm_pixels, lsm.w, lsm.pad_x(x), lsm.pad_y(y), radius, win_pix_count);
                float l_var = integral_mean(lsq_pixels, lsq.w, lsq.pad_x(x), lsq.pad_y(y), radius, win_pix_count) - (l_mean * l_mean);
                for(int d = min_d; d <= max_d; ++d){
                    int rx = x + (d*d_dir);
                    //Calculate right mean, calculate zncc score
                    float r_mean = integral_mean(rsm_pixels, rsm.w, rsm.pad_x(rx), rsm.pad_y(y), radius, win_pix_count);
                    float zncc = calculate_zncc_sse_128(
                        d_map, lsq, rsq, 
                        l_pixels, r_pixels,
                        lsq_pixels, rsq_pixels,
                        x, rx, y, radius, 
                        l_mean, r_mean, l_mean, win_pix_count
                    );
                    if(zncc > max_zncc){
                        max_zncc = zncc;
                        zncc_max_d = d;
                    }
                }
                d_map_pixels[y * d_map.w + x] = zncc_max_d;
            }
        }
    }
#endif

    //Calculates the disparity using the ZNCC algo. Calculates disparity shift from left image to right image, storing values in map image.
    bool calculate_disparity_map(
        int radius, int min_d, int max_d, const bool left_to_right,
        Image& lsm_map, Image& rsm_map, Image& lsq_map, Image& rsq_map,
        Image& left, Image& right, Image& map, ThreadPool& thread_pool, std::string scope_tag
    ){
    #if defined(__SSE4_1__)
        if(left.w == right.w && left.h == right.h && left.format == ImageFormat::GRAY && right.format == ImageFormat::GRAY && radius > 0){
            mp::ScopeTimer exec_timer("mp::zncc_c_multi_thread::calculate_disparity_map_" + scope_tag);
            //Allocate disparity map
            map.init<uint8_t>(left.w, left.h, ImageFormat::GRAY);
            const int d_dir = left_to_right ? -1 : 1;
            const int win_pix_count = ((radius << 1) + 1) * ((radius << 1) + 1);
            //Threading strat => Split the work into Chunks of X rows.
            queue_linear_work(
                thread_pool, left.h, calculate_disparity_map_work_chunk,
                radius, min_d, max_d, d_dir, win_pix_count,
                std::ref(left), std::ref(right), std::ref(map), std::ref(lsm_map), 
                std::ref(rsm_map), std::ref(lsq_map), std::ref(rsq_map)
            );
            //Wait for work to finish
            thread_pool.wait_for_work();
            return true;
        }
    #endif
        return false;
    }

    //Maps the disparity value scale to grayscale
    uint8_t grayscale_disparity(const uint8_t min_d, const uint8_t max_d, const uint8_t value){
        if(value <= min_d){
            return 0;
        }
        return 255.f * (value - min_d) / (max_d - min_d); 
    }

    //Calculate middle value using histogram
    uint8_t calculate_window_non_zero_middle(int x, int y, int radius, Image& image){
        //Assume grayscale
        uint8_t* pixels = image.data<uint8_t>();
        int count = 0;
        std::array<int, 256> histogram = {};
        //Count the values with in the buckets
        for(int yr = -radius; yr <= radius; ++yr){
            int cy = image.clamp_y(y + yr) * image.w;
            for(int xr = -radius; xr <= radius; ++xr){
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
        int threshold_value, int radius, int min_d, int max_d, Image& left_disparity, Image& right_disparity, Image& pp_disparity,
        const int chunk_start, const int chunk_end
    ){
        uint8_t * l_pixels = left_disparity.data<uint8_t>();
        uint8_t * r_pixels = right_disparity.data<uint8_t>();
        uint8_t * pp_pixels = pp_disparity.data<uint8_t>();
        for(int y = chunk_start; y < chunk_end; ++y){
            for(int x = 0; x < left_disparity.w; ++x){ 
                //Pull disparity L=>R and R=>L
                uint8_t disparity_value_l = l_pixels[y * left_disparity.w + x];
                uint8_t disparity_value_r = 0;
                if(x - disparity_value_l >= 0){
                    disparity_value_r = r_pixels[y * left_disparity.w + x - disparity_value_l];
                }
                uint8_t final_value = disparity_value_l;
                //Cross-check and occulsion
                if(abs(disparity_value_l - disparity_value_r) > threshold_value || final_value == 0){
                    //Get the window middle value as filler
                    final_value = calculate_window_non_zero_middle(x, y, radius, left_disparity);
                }
                pp_pixels[y * left_disparity.w + x] = grayscale_disparity(min_d, max_d, final_value);
            }
        }
    }

    //Cross-checks the two disparity maps against each other
    bool cross_check_occulsion_disparity_maps(int threshold_value, int radius, int min_d, int max_d, Image& left_disparity, Image& right_disparity, Image& pp_disparity, ThreadPool& thread_pool){
        if(
            left_disparity.format == ImageFormat::GRAY && right_disparity.format == ImageFormat::GRAY &&
            left_disparity.w == right_disparity.w && left_disparity.h == right_disparity.h
        ){
            mp::ScopeTimer exec_timer("mp::zncc_c_multi_thread::cross_check_occulsion_disparity_maps");
            //reserve space for the post processed map
            pp_disparity.init<uint8_t>(left_disparity.w, left_disparity.h, ImageFormat::GRAY);
            //Threading strat => Split the work into Chunks of X rows.
            queue_linear_work(
                thread_pool, left_disparity.h, cross_check_occulsion_disparity_maps_work_chunk,
                threshold_value, radius, min_d, max_d, 
                std::ref(left_disparity), std::ref(right_disparity), std::ref(pp_disparity)
            );
            //Wait for work to finish
            thread_pool.wait_for_work();
            return true;
        }
        return false;
    }

}