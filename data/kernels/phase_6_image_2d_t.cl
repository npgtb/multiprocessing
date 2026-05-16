
//Resize the image by factor. Takes the nth row and column approach. Expects a RGBA format image.
__kernel void resize_image( __read_only image2d_t original, __write_only image2d_t downscaled, const int factor){
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    int2 downscaled_coordinate = (int2)(x, y);
    int2 original_coordinate = (int2)(x * factor, y * factor);
    const sampler_t psampler =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP |
        CLK_FILTER_NEAREST;
    uint4 pixel = read_imageui(
        original, psampler, original_coordinate
    );
    write_imageui(downscaled, downscaled_coordinate, pixel);
}

//Grayscales the image. Expects a RGBA format image and turns image to GRAY format.
__kernel void grayscale_image(__read_only image2d_t original, __write_only image2d_t grayscaled){
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    int2 coordinate = (int2)(x, y);
    const sampler_t psampler =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP |
        CLK_FILTER_NEAREST;
    uint4 pixel = read_imageui(
        original, psampler, coordinate
    );
    float3 weights = (float3)(0.2126f, 0.7152f, 0.0722f);
    float gray_float = dot(convert_float3(pixel.xyz), weights);
    uint4 gray_pixel = (uint4)((uchar)(gray_float + 0.5f), 0, 0, 0);
    write_imageui(grayscaled, coordinate, gray_pixel);
}

//Calculates mean from the window in the given image. Expects grayscale image
float calculate_window_mean(
    int x, int y, int radius, __read_only image2d_t img, 
    const int win_pix_count, const sampler_t psampler
){
    float sum = 0;
    //Go trough the window
    for(int yr = -radius; yr <= radius; ++yr){
        int cy = y + yr;
        for(int xr = -radius; xr <= radius; ++xr){
            int cx = x + xr;
            sum += (read_imageui(img, psampler, (int2)(cx, cy))).x;
        }
    }
    return sum / win_pix_count;
}

//Calculates the ZNCC trough upper and lower sums. Expects grayscaled images
float calculate_zncc(
    int x, int rx, int y, int radius, float lmean, float rmean,
    __read_only image2d_t limg, __read_only image2d_t rimg, const sampler_t psampler
){
    float upper = 0.f;
    float lower_l = 0.f, lower_r = 0.f;
    for(int yr = -radius; yr <= radius; ++yr){
        int cy = y + yr;
        for(int xr = -radius; xr <= radius; ++xr){
            int lcx = x + xr;
            int rcx = rx + xr;
            uint4 left_pixel = read_imageui(limg, psampler, (int2)(lcx, cy));
            uint4 right_pixel = read_imageui(rimg, psampler, (int2)(rcx, cy));
            //Normalize brightness with means
            float l_diff = ((float)left_pixel.x) - lmean;
            float r_diff = ((float)right_pixel.x) - rmean;
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
    __read_only image2d_t limg, __read_only image2d_t rimg, __write_only image2d_t d_map,
    const int radius, const int min_d, const int max_d, 
    const int d_dir
){
    //Global position in the image
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);
    //Pixel sampler
    const sampler_t psampler =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP |
        CLK_FILTER_NEAREST;

    //Pixel count in a window
    const int win_pix_count = (((radius << 1) + 1) * ((radius << 1) + 1));

    float max_zncc = -1.f; 
    uchar zncc_max_d = 0;
    //Calculate Left mean
    float left_mean = calculate_window_mean(gx, gy, radius, limg, win_pix_count, psampler);
    for(int d = min_d; d <= max_d ; ++d){
        int rx = gx + (d * d_dir);
        //Right mean into, zncc calculation
        float right_mean = calculate_window_mean(rx, gy, radius, rimg, win_pix_count, psampler);
        float zncc = calculate_zncc(gx, rx, gy, radius, left_mean, right_mean, limg, rimg, psampler);
        //Check if we have new highscore
        if(zncc > max_zncc){
            max_zncc = zncc;
            zncc_max_d = d;
        }
    }
    write_imageui(d_map, (int2)(gx, gy), (uint4)(zncc_max_d, 0, 0, 0));
}

//Maps the disparity value scale to grayscale
uchar grayscale_disparity(const uchar min_d, const uchar max_d, const uchar value){
    if(value <= min_d){
        return 0;
    }
    return (uchar)((255 * (value - min_d)) / (max_d - min_d));
}

//Calculate middle value using histogram
uchar calculate_window_non_zero_middle(int x, int y, int radius, __read_only image2d_t img, const sampler_t psampler){
    int count = 0;
    ushort histogram[256] = {0};
    //Count the values with in the buckets
    for(int yr = -radius; yr <= radius; ++yr){
        int cy = y + yr;
        for(int xr = -radius; xr <= radius; ++xr){
            int cx = x + xr;
            uint4 pixel_value = read_imageui(img, psampler, (int2)(cx, cy));
            if(pixel_value.x > 0){
                histogram[pixel_value.x]++;
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
    __read_only image2d_t limg, __read_only image2d_t rimg, __write_only image2d_t post_processed,
    int threshold_value, int radius, int min_d, int max_d
){
    //Global position in the image
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    //pixel sampler
    const sampler_t psampler =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP |
        CLK_FILTER_NEAREST;

    //Pull disparity L=>R and R=>L
    uint4 disparity_value_l = read_imageui(limg, psampler, (int2)(x, y));
    uint4 disparity_value_r = read_imageui(
        rimg, psampler, (int2)(x - disparity_value_l.x, y)
    );
    
    //Cross-check and occulsion
    uchar final_value = disparity_value_l.x;
    if(abs(disparity_value_l.x - disparity_value_r.x) > threshold_value || final_value == 0){
        //Get the window middle value as filler
        final_value = calculate_window_non_zero_middle(x, y, radius, limg, psampler);
    }
    write_imageui(post_processed, (int2)(x, y), (uint4)(grayscale_disparity(min_d, max_d, final_value), 0, 0, 0));
}