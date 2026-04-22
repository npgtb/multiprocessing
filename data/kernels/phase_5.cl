
//Resize the image by factor. Takes the nth row and column approach. Expects a RGBA format image.
__kernel void resize_image(__global uint* original, __global uint* downscaled, const int factor, const int original_width){
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int new_width = get_global_size(0);
    downscaled[y * new_width + x] = original[(y*factor) * original_width + (x * factor)];
}

//Grayscales the image. Expects a RGBA format image and turns image to GRAY format.
__kernel void grayscale_image(__global uint* original, __global uchar* grayscaled){
    const int red_shift = 24, green_shift = 16, blue_shift = 8;
    const float red_modifier = 0.2126, green_modifier = 0.7152, blue_modifier = 0.0722;
    const uint  mask = 0xFF;
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int width = get_global_size(0);
    const int i = y * width + x;
    grayscaled[i] = (uchar)
        (
            ((original[i] >> red_shift) & mask) * red_modifier + 
            ((original[i] >> green_shift) & mask) * green_modifier +
            ((original[i] >> blue_shift) & mask) * blue_modifier
            + 0.5f
        );
}

//Calculates mean from the window in the given image. Expects grayscale image
float calculate_window_mean(int x, int y, int radius,  __global uchar* pixels, int width, int height){
    int window_size = (radius * 2 + 1);
    int count = window_size * window_size;
    float sum = 0;
    //Go trough the window
    for(int yr = -radius; yr <= radius; ++yr){
        int cy = clamp((y + yr), 0, height - 1) * width;
        for(int xr = -radius; xr <= radius; ++xr){
            int cx = clamp((x + xr), 0, width - 1);
            sum += pixels[cy + cx];
        }
    }
    return sum / count;
}

//Calculates the ZNCC trough upper and lower sums. Expects grayscaled images
float calculate_zncc(int x, int rx, int y, int radius, float lmean, float rmean, __global uchar* left, __global uchar* right,  int width, int height){
    float upper = 0.f;
    float lower_l = 0.f, lower_r = 0.f;
    //Go trough the window
    for(int yr = -radius; yr <= radius; ++yr){
        int cy = clamp((y + yr), 0, height - 1) * width;
        for(int xr = -radius; xr <= radius; ++xr){
            int lcx = clamp((x + xr), 0, width - 1);
            int rcx = clamp((rx + xr), 0, width - 1);
            //Normalize brightness with means
            float l_diff = ((float)left[cy + lcx]) - lmean;
            float r_diff = ((float)right[cy + rcx]) - rmean;
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
    return 0;
}

//Calculates the disparity using the ZNCC algo. Calculates disparity shift from left image to right image, storing values in map image.
__kernel void calculate_disparity_map(
    __global uchar* left, __global uchar* right, __global uchar* map,
    const int window_radius, const int min_disparity, const int max_disparity, 
    const int disparity_direction
){
    //pull globals
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);
    const int width = get_global_size(0);
    const int height = get_global_size(1);

    float max_zncc = -1.f; 
    uchar zncc_max_disparity = 0;
    //Calculate Left mean
    float left_mean = calculate_window_mean(gx, gy, window_radius, left, width, height);
    for(int d = min_disparity; d <= max_disparity ; ++d){
        int rx = gx + (d * disparity_direction);
        //Calculate right mean and zncc score
        float right_mean = calculate_window_mean(rx, gy, window_radius, right,width, height);
        float zncc = calculate_zncc(gx, rx, gy, window_radius, left_mean, right_mean, left, right,width, height);
        if(zncc > max_zncc){
            max_zncc = zncc;
            zncc_max_disparity = d;
        }
    }
    map[gy * width + gx] = zncc_max_disparity;
}

//Maps the disparity value scale to grayscale
uchar grayscale_disparity(
    const uchar min_disparity, const uchar max_disparity, const uchar value
){
    if(value <= min_disparity){
        return 0;
    }
    return (uchar)((255 * (value - min_disparity)) / (max_disparity - min_disparity));
}

//Calculate middle value using histogram
uchar calculate_window_non_zero_middle(int x, int y, int radius, __global uchar* pixels, const int width, const int height){
    int count = 0;
    int histogram[256] = {0};
    //Count the values with in the buckets
    for(int yr = -radius; yr <= radius; ++yr){
        int cy = clamp(y + yr, 0, height-1) * width;
        for(int xr = -radius; xr <= radius; ++xr){
            //Edge handling => nearest valid pixel
            int cx = clamp(x + xr, 0, width-1);
            uchar pixel_value = pixels[cy + cx];
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

//Cross-checks the two disparity maps against each other
__kernel void cross_check_occulsion_disparity_maps(
    __global uchar* left_pixels, __global uchar* right_pixels, __global uchar* pp_pixels,
    int threshold_value, int window_radius, int min_disparity, int max_disparity
){
    //pull globals
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int width = get_global_size(0);
    const int height = get_global_size(1);
    //Get disparity from L=>R
    uchar disparity_value_l = left_pixels[y * width + x];
    uchar disparity_value_r = 0;
    if(x - disparity_value_l >= 0){
        //If possible get disparity R=>L
        disparity_value_r = right_pixels[(y * width + x) - disparity_value_l];
    }
    //Cross-check and occulsion
    uchar final_value = disparity_value_l;
    if(abs(disparity_value_l - disparity_value_r) > threshold_value || final_value == 0){
        //Get the window middle value as filler
        final_value = calculate_window_non_zero_middle(x, y, window_radius, left_pixels, width, height);
    }
    pp_pixels[y * width + x] = grayscale_disparity(min_disparity, max_disparity, final_value);
}