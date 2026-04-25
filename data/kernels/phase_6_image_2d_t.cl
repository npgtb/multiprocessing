
//Resize the image by factor. Takes the nth row and column approach. Expects a RGBA format image.
__kernel void resize_image( __read_only image2d_t original, __write_only image2d_t downscaled, const int factor){
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    int2 downscaled_coordinate = (int2)(x, y);
    int2 original_coordinate = (int2)(x * factor, y * factor);
    const sampler_t pixel_sampler =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP |
        CLK_FILTER_NEAREST;
    uint4 pixel = read_imageui(
        original, pixel_sampler, original_coordinate
    );
    write_imageui(downscaled, downscaled_coordinate, pixel);
}

//Grayscales the image. Expects a RGBA format image and turns image to GRAY format.
__kernel void grayscale_image(__read_only image2d_t original, __write_only image2d_t grayscaled){
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    int2 coordinate = (int2)(x, y);
    const sampler_t pixel_sampler =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP |
        CLK_FILTER_NEAREST;
    uint4 pixel = read_imageui(
        original, pixel_sampler, coordinate
    );
    float3 weights = (float3)(0.2126f, 0.7152f, 0.0722f);
    float gray_float = dot(convert_float3(pixel.xyz), weights);
    uint4 gray_pixel = (uint4)((uchar)(gray_float + 0.5f), 0, 0, 0);
    write_imageui(grayscaled, coordinate, gray_pixel);
}

//Calculates mean from the window in the given image. Expects grayscale image
float calculate_window_mean(int x, int y, int radius, __read_only image2d_t pixels, int width, int height, const sampler_t pixel_sampler){
    int window_size = (radius * 2 + 1);
    int count = window_size * window_size;
    float sum = 0;
    //Go trough the window
    for(int yr = -radius; yr <= radius; ++yr){
        int cy = clamp((y + yr), 0, height - 1);
        for(int xr = -radius; xr <= radius; ++xr){
            int cx = clamp((x + xr), 0, width - 1);
            int2 coordinate = (int2)(cx, cy);
            uint4 pixel = read_imageui(pixels, pixel_sampler, coordinate);
            sum += pixel.x;
        }
    }
    return sum / count;
}

//Calculates the ZNCC trough upper and lower sums. Expects grayscaled images
float calculate_zncc(
    int x, int rx, int y, int radius, float lmean, float rmean,
    __read_only image2d_t left, __read_only image2d_t right,  int width, int height,
    const sampler_t pixel_sampler
){
    float upper = 0.f;
    float lower_l = 0.f, lower_r = 0.f;
    for(int yr = -radius; yr <= radius; ++yr){
        int cy = clamp((y + yr), 0, height - 1);
        for(int xr = -radius; xr <= radius; ++xr){
            int lcx = clamp((x + xr), 0, width - 1);
            int rcx = clamp((rx + xr), 0, width - 1);
            uint4 left_pixel = read_imageui(left, pixel_sampler, (int2)(lcx, cy));
            uint4 right_pixel = read_imageui(right, pixel_sampler, (int2)(rcx, cy));
            //Normalize brightness with means
            float l_diff = ((float)left_pixel.x) - lmean;
            float r_diff = ((float)right_pixel.x) - rmean;
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
    __read_only image2d_t left, __read_only image2d_t right, __write_only image2d_t map,
    const int window_radius, const int min_disparity, const int max_disparity, 
    const int disparity_direction
){
    //pull globals
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);
    const int width = get_global_size(0);
    const int height = get_global_size(1);
    const sampler_t pixel_sampler =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP |
        CLK_FILTER_NEAREST;

    float max_zncc = -1.f; 
    uchar zncc_max_disparity = 0;

    //Calculate Left mean
    float left_mean = calculate_window_mean(gx, gy, window_radius, left, width, height, pixel_sampler);
    for(int d = min_disparity; d <= max_disparity ; ++d){
        int rx = gx + (d * disparity_direction);
        //Right mean into, zncc calculation
        float right_mean = calculate_window_mean(rx, gy, window_radius, right,width, height, pixel_sampler);
        float zncc = calculate_zncc(gx, rx, gy, window_radius, left_mean, right_mean, left, right,width, height, pixel_sampler);
        if(zncc > max_zncc){
            max_zncc = zncc;
            zncc_max_disparity = d;
        }
    }
    write_imageui(map, (int2)(gx, gy), (uint4)(zncc_max_disparity, 0, 0, 0));
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
uchar calculate_window_non_zero_middle(
    int x, int y, int radius, __read_only image2d_t pixels, const int width, const int height, const sampler_t pixel_sampler
){
    int count = 0;
    int histogram[256] = {0};
    //Count the values with in the buckets
    for(int yr = -radius; yr <= radius; ++yr){
        int cy = clamp(y + yr, 0, height-1);
        for(int xr = -radius; xr <= radius; ++xr){
            int cx = clamp(x + xr, 0, width-1);
            uint4 pixel_value = read_imageui(pixels, pixel_sampler, (int2)(cx, cy));
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
    __read_only image2d_t left_pixels, __read_only image2d_t right_pixels, __write_only image2d_t pp_pixels,
    int threshold_value, int window_radius, int min_disparity, int max_disparity
){
    //pull globals
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int width = get_global_size(0);
    const int height = get_global_size(1);
    const sampler_t pixel_sampler =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP |
        CLK_FILTER_NEAREST;

    //Get disparity from L=>R
    uint4 disparity_value_l = read_imageui(left_pixels, pixel_sampler, (int2)(x, y));
    uint4 disparity_value_r = 0;
    if(x - disparity_value_l.x >= 0){
        //If possible get disparity R=>L
        disparity_value_r = read_imageui(
            right_pixels, pixel_sampler, (int2)(x - disparity_value_l.x, y)
        );
    }
    //Cross-check and occulsion
    uchar final_value = disparity_value_l.x;
    if(abs(disparity_value_l.x - disparity_value_r.x) > threshold_value || final_value == 0){
        //Get the window middle value as filler
        final_value = calculate_window_non_zero_middle(x, y, window_radius, left_pixels, width, height, pixel_sampler);
    }
    write_imageui(pp_pixels, (int2)(x, y), (uint4)(grayscale_disparity(min_disparity, max_disparity, final_value), 0, 0, 0));
}