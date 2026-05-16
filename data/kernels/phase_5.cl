
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
float calculate_window_mean(
    int x, int y, int radius, const int win_pix_count,
    __global uchar* img, int imgw, int imgh
){
    float sum = 0;
    //Go trough the window
    for(int yr = -radius; yr <= radius; ++yr){
        int cy = clamp((y + yr), 0, imgh - 1) * imgw;
        for(int xr = -radius; xr <= radius; ++xr){
            int cx = clamp((x + xr), 0, imgw - 1);
            sum += img[cy + cx];
        }
    }
    return sum / win_pix_count;
}

//Calculates the ZNCC trough upper and lower sums. Expects grayscaled images
float calculate_zncc(
    int x, int rx, int y, int radius, float lmean, float rmean,
    __global uchar* limg, __global uchar* rimg,  int imgw, int imgh
){
    float upper = 0.f;
    float lower_l = 0.f, lower_r = 0.f;
    //Go trough the window
    for(int yr = -radius; yr <= radius; ++yr){
        int cy = clamp((y + yr), 0, imgh - 1) * imgw;
        for(int xr = -radius; xr <= radius; ++xr){
            int lcx = clamp((x + xr), 0, imgw - 1);
            int rcx = clamp((rx + xr), 0, imgw - 1);
            //Normalize brightness with means
            float l_diff = ((float)limg[cy + lcx]) - lmean;
            float r_diff = ((float)rimg[cy + rcx]) - rmean;
            //Adjust upper and lower sums
            upper += l_diff * r_diff;
            lower_l += (l_diff * l_diff);
            lower_r += (r_diff * r_diff);
        }
    }
    float divider = sqrt(lower_l * lower_r);
    if(divider != 0){
        return upper / divider;
    }
    return 0;
}

//Calculates the disparity using the ZNCC algo. Calculates disparity shift from left image to right image, storing values in map image.
__kernel void calculate_disparity_map(
    __global uchar* limg, __global uchar* rimg, __global uchar* map,
    const int radius, const int min_d, const int max_d, 
    const int d_dir
){
    //Global position and dimensions
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);
    const int imgw = get_global_size(0);
    const int imgh = get_global_size(1);

    //Pixel count in a window
    const int win_pix_count = (((radius << 1) + 1) * ((radius << 1) + 1));

    float max_zncc = -1.f; 
    uchar zncc_max_d = 0;
    //Calculate Left mean
    float left_mean = calculate_window_mean(gx, gy, radius, win_pix_count, limg, imgw, imgh);
    for(int d = min_d; d <= max_d ; ++d){
        int rx = gx + (d * d_dir);
        //Calculate right mean and zncc score
        float right_mean = calculate_window_mean(rx, gy, radius, win_pix_count, rimg, imgw, imgh);
        float zncc = calculate_zncc(gx, rx, gy, radius, left_mean, right_mean, limg, rimg, imgw, imgh);
        //Check if we have new highscore
        if(zncc > max_zncc){
            max_zncc = zncc;
            zncc_max_d = d;
        }
    }
    map[gy * imgw + gx] = zncc_max_d;
}

//Maps the disparity value scale to grayscale
uchar grayscale_disparity(const uchar min_d, const uchar max_d, const uchar value){
    if(value <= min_d){
        return 0;
    }
    return (uchar)((255 * (value - min_d)) / (max_d - min_d));
}

//Calculate middle value using histogram
uchar calculate_window_non_zero_middle(int x, int y, int radius, __global uchar* img, const int imgw, const int imgh){
    int count = 0;
    ushort histogram[256] = {0};
    //Count the values with in the buckets
    for(int yr = -radius; yr <= radius; ++yr){
        int cy = clamp(y + yr, 0, imgh-1) * imgw;
        for(int xr = -radius; xr <= radius; ++xr){
            int cx = clamp(x + xr, 0, imgw-1);
            uchar pixel_value = img[cy + cx];
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
        ushort * histogram_value = &histogram[1];
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
    __global uchar* limg, __global uchar* rimg, __global uchar* post_processed,
    int threshold_value, int radius, int min_d, int max_d
){
    //Global position and dimensions
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int imgw = get_global_size(0);
    const int imgh = get_global_size(1);
    //Get disparity from L=>R and R=>L
    uchar disparity_value_l = limg[y * imgw + x];
    uchar disparity_value_r = 0;
    if(x - disparity_value_l >= 0){
        disparity_value_r = rimg[(y * imgw + x) - disparity_value_l];
    }
    //Cross-check and occulsion
    uchar final_value = disparity_value_l;
    if(abs(disparity_value_l - disparity_value_r) > threshold_value || final_value == 0){
        //Get the window middle value as filler
        final_value = calculate_window_non_zero_middle(x, y, radius, limg, imgw, imgh);
    }
    post_processed[y * imgw + x] = grayscale_disparity(min_d, max_d, final_value);
}