#include <cmath>
#include <array>
#include <scope_timer.h>
#include <cpu_workloads/phase_3.h>

namespace mp::cpu_workloads::phase_3{

    //Resize the image by factor. Takes the nth row and column approach. Expects a RGBA format image.
    bool resize_image(Image& input, Image& output, const int factor){
        if(input.format == ImageFormat::RGBA){
            mp::ScopeTimer exec_timer("mp::zncc_c_single_thread::resize_image");
            //Calculate new dimensions and allocate memory
            const int new_width = input.w / factor;
            const int new_height = input.h / factor;
            uint32_t * original_image = static_cast<uint32_t*>(input.pixels);
            uint32_t * downsized_image = static_cast<uint32_t*>(malloc(new_width * new_height * sizeof(uint32_t)));
            if(downsized_image){
                for(int j = 0; j < new_height; ++j){
                    for(int i = 0; i < new_width; ++i){
                        //Strategy: every factor'th column and row
                        downsized_image[j * new_width + i] = original_image[(j*factor) * input.w + (i * factor)];
                    }
                }
                output.set(downsized_image, new_width, new_height, ImageFormat::RGBA);
                return true;
            }
        }
        return false;
    }

    //Grayscales the image. Expects a RGBA format image and turns image to GRAY format.
    bool grayscale_image(Image& image){
        if(image.format == ImageFormat::RGBA){
            mp::ScopeTimer exec_timer("mp::zncc_c_single_thread::grayscale_image");
            //Set modifiers and allocate memory
            const int memory_size = image.w * image.h;
            constexpr int red_shift = 24, green_shift = 16, blue_shift = 8;
            constexpr float red_modifier = 0.2126, green_modifier = 0.7152, blue_modifier = 0.0722;
            constexpr uint8_t mask = 0xFF;
            uint8_t * grayscaled_pixels = static_cast<uint8_t*>(malloc(memory_size * sizeof(uint8_t)));
            if(grayscaled_pixels){
                uint32_t * rgba_pixels = static_cast<uint32_t*>(image.pixels);
                for(int i = 0; i < memory_size; ++i){
                    //Apply the grayscale formula to the image pixels
                    grayscaled_pixels[i] = static_cast<uint8_t>(
                        std::round(
                            ((rgba_pixels[i] >> red_shift) & mask) * red_modifier + 
                            ((rgba_pixels[i] >> green_shift) & mask) * green_modifier +
                            ((rgba_pixels[i] >> blue_shift) & mask) * blue_modifier
                        )
                    );
                }
                image.set(grayscaled_pixels, image.w, image.h, ImageFormat::GRAY);
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
            int cy = image.clamp_y(y + yr) * image.w;
            for(int xr = -radius; xr <= radius; ++xr){
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
                int lcx = left.clamp_x(x + xr);
                int rcx = right.clamp_x(rx + xr);
                //Normalize brightness with means
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

    //Calculates the disparity using the ZNCC algo. Calculates disparity shift from left image to right image, storing values in map image.
    bool calculate_disparity_map(const int window_radius, const int min_disparity, const int max_disparity, const bool left_to_right, Image& left, Image& right, Image& map, std::string scope_tag){
        if(left.w == right.w && left.h == right.h && left.format == ImageFormat::GRAY && right.format == ImageFormat::GRAY && window_radius > 0){
            mp::ScopeTimer exec_timer("mp::zncc_c_single_thread::calculate_disparity_map_" + scope_tag);
            //Allocate disparity map
            uint8_t* disparity_map_pixels = static_cast<uint8_t*>(malloc(left.w * left.h));
            if(disparity_map_pixels){
                map.set(disparity_map_pixels, left.w, left.h, ImageFormat::GRAY);
                int disparity_direction = 1;
                if(left_to_right){
                    disparity_direction = -1;
                }
                for(int y = 0; y < left.h; ++y){
                    for(int x = 0; x < left.w; ++x){
                        float max_zncc = -1.f; 
                        uint8_t zncc_max_disparity = 0;
                        //Calculate Left mean
                        float left_mean = calculate_window_mean(x, y, window_radius, left);
                        for(int d = min_disparity; d <= max_disparity; ++d){
                            int rx = x + (d * disparity_direction);
                            //Calculate right mean and zncc score
                            float right_mean = calculate_window_mean(rx, y, window_radius, right);
                            float zncc = calculate_zncc(x, rx, y, window_radius, left_mean, right_mean, left, right);
                            if(zncc > max_zncc){
                                max_zncc = zncc;
                                zncc_max_disparity = d;
                            }
                        }
                        disparity_map_pixels[y * map.w + x] = zncc_max_disparity;
                    }
                }
                return true;
            }
        }
        return false;
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

    //Maps the disparity value scale to grayscale
    uint8_t grayscale_disparity(
        const uint8_t min_disparity, const uint8_t max_disparity, const uint8_t value
    ){
        if(value <= min_disparity){
            return 0;
        }
        return 255.f * (value - min_disparity) / (max_disparity - min_disparity); 
    }

    //Cross-checks the two disparity maps against each other
    bool cross_check_occulsion_disparity_maps(
        const int threshold_value, const int window_radius, const int min_disparity, const int max_disparity, 
        Image& left_disparity, Image& right_disparity, Image& pp_disparity
    ){
        if(
            left_disparity.format == ImageFormat::GRAY && right_disparity.format == ImageFormat::GRAY &&
            left_disparity.w == right_disparity.w && left_disparity.h == right_disparity.h
        ){
            mp::ScopeTimer exec_timer("mp::zncc_c_single_thread::cross_check_occulsion_disparity_maps");
            //reserve space for the post processed map
            uint8_t * pp_pixels = static_cast<uint8_t*>(malloc(left_disparity.w * left_disparity.h));
            if(pp_pixels){
                pp_disparity.set(pp_pixels, left_disparity.w, left_disparity.h, ImageFormat::GRAY);
                uint8_t * left_pixels = static_cast<uint8_t*>(left_disparity.pixels);
                uint8_t * right_pixels = static_cast<uint8_t*>(right_disparity.pixels);
                for(int y = 0; y < left_disparity.h; ++y){
                    for(int x = 0; x < left_disparity.w; ++x){ 
                        //Pull disparity L=>R and R=>L
                        uint8_t disparity_value_l = left_pixels[y * left_disparity.w + x];
                        uint8_t disparity_value_r = 0;
                        if(x - disparity_value_l >= 0){
                            disparity_value_r = right_pixels[y * left_disparity.w + x - disparity_value_l];
                        }
                        uint8_t final_value = disparity_value_l;
                        //Cross-check and occulsion
                        if(abs(disparity_value_l - disparity_value_r) > threshold_value || final_value == 0){
                            //Get the window middle value as filler
                            final_value = calculate_window_non_zero_middle(x, y, window_radius, left_disparity);
                        }
                        pp_pixels[y * left_disparity.w + x] = grayscale_disparity(min_disparity, max_disparity, final_value);
                    }
                }
                return true;
            }
        }
        return false;
    }
}