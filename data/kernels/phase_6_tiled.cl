
//Resize the image by factor. Takes the nth row and column approach. Expects a RGBA format image.
__kernel void resize_image(__global uint* original, __global uint* downscaled, const int factor, const int original_width){
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int new_width = get_global_size(0);
    downscaled[y * new_width + x] = original[(y*factor) * original_width + (x * factor)];
}

//Grayscales the image. Expects a RGBA format image and turns image to GRAY format.
__kernel void grayscale_image(__global uint* original, __global uchar* grayscaled){
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int width = get_global_size(0);
    uchar4 original_pixel = as_uchar4(original[(y * width + x)]);
    float3 weights = (float3)(0.2126f, 0.7152f, 0.0722f);
    grayscaled[y * width + x] = (uchar) (dot(convert_float3(original_pixel.wzy), weights) + (.5f));
}

//Calculates mean from the window in the given image. Expects grayscale image
inline float calculate_window_mean(int x, int y, int radius, __local uchar* pixels, const int width){
    float sum = 0;
    //Go trough the window
    for(int yr = -radius; yr <= radius; ++yr){
        __local uchar * pixel = &pixels[((y + yr) * width) + (x - radius)];
        for(int xr = -radius; xr <= radius; ++xr){
            sum += *pixel++;
        }
    }
    return sum / (((radius << 1) + 1) * ((radius << 1) + 1));
}

//Calculates the ZNCC trough upper and lower sums. Expects grayscaled images
inline float calculate_zncc(
    int lx, int lxr, int ly, int radius, float lmean, float rmean, 
    __local uchar* left, __local uchar* right, const int left_width, const int right_width
){
    float upper = 0.f;
    float lower_l = 0.f, lower_r = 0.f;
    //Go trough the window
    for(int yr = -radius; yr <= radius; ++yr){
        __local uchar * left_pixel = &left[((ly + yr) * left_width) + (lx - radius)];
        __local uchar * right_pixel = &right[((ly + yr) * right_width) + (lxr - radius)];
        for(int xr = -radius; xr <= radius; ++xr){
            //Normalize brightness with means
            float l_diff = (float)(*left_pixel++) - lmean;
            float r_diff = (float)(*right_pixel++) - rmean;
            //Adjust upper and lower sums
            upper += l_diff * r_diff;
            lower_l += (l_diff * l_diff);
            lower_r += (r_diff * r_diff);
        }
    }
    float result = upper / (sqrt(lower_l * lower_r));
    return select(0.f, result, isfinite(result));
}

//Calculates the disparity using the ZNCC algo. Calculates disparity shift from left image to right image, storing values in map image.
__kernel void calculate_disparity_map(
    __global uchar* left, __global uchar* right, __global uchar* map,
    const int window_radius, const int min_disparity, const int max_disparity, const int disparity_direction,
    const int width, const int height, __local uchar * left_tile, __local uchar * right_tile
){
    //pull locals
    const int local_width = get_local_size(0);
    const int local_height = get_local_size(1);

    //Calculate tile dimensions
    const int tile_left_width = (window_radius << 1) + local_width;
    const int tile_height = (window_radius << 1) + local_height;
    const int tile_right_width = local_width +  (window_radius << 1) + (max_disparity - min_disparity);

    const int group_base_x = local_width * get_group_id(0);
    const int group_base_y = local_height * get_group_id(1);
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    //Get tiled pixels
    for(int tile_y = ly; tile_y < tile_height; tile_y += local_height){      
        int image_y = clamp(group_base_y + tile_y - window_radius, 0, height - 1);  
        //Load left tile
        for(int tile_x = lx; tile_x < tile_left_width; tile_x += local_width){
            int image_x = clamp(group_base_x + tile_x - window_radius, 0, width - 1);
            left_tile[tile_y * tile_left_width + tile_x] = left[image_y * width + image_x];
        }
        //Load right tile
        for(int tile_x = lx; tile_x < tile_right_width; tile_x += local_width){
            int shift_x = tile_x - window_radius;
            //Depending on the disparity direction, we either have massive extra chunk 
            //at the left side of the tile or its at the right side of the tile
            shift_x -= disparity_direction == -1 ? max_disparity : 0;
            //We shift to the left/right by disparity range + window_radius
            int image_x = clamp(group_base_x + shift_x, 0, width - 1);
            right_tile[tile_y * tile_right_width + tile_x] = right[image_y * width + image_x];
        }
    }

    //Wait for whole group finish pulling pixel data
    barrier(CLK_LOCAL_MEM_FENCE);

    const int gx = get_global_id(0);
    const int gy = get_global_id(1);
    //Dont calculate padded pixels
    if((gx >= width || gy >= height)){
        return;
    }
    //Tiling caused offset coordinates
    const int lx_halo = lx + window_radius;
    const int rx_halo = lx_halo + (disparity_direction == -1 ? max_disparity : 0);
    const int ly_halo = ly + window_radius;

    float max_zncc = -1.f; 
    uchar zncc_max_disparity = 0;
    //Calculate Left mean
    float left_mean = calculate_window_mean(lx_halo, ly_halo, window_radius, left_tile, tile_left_width);
    for(uchar d = min_disparity; d <= max_disparity; ++d){
        int rx = rx_halo + (d * disparity_direction);
        //Calculate right mean and zncc score
        float right_mean = calculate_window_mean(rx, ly_halo, window_radius, right_tile, tile_right_width);
        float zncc = calculate_zncc(lx_halo, rx, ly_halo, window_radius, left_mean, right_mean, left_tile, right_tile, tile_left_width, tile_right_width);
        bool better_disparity = zncc > max_zncc;
        zncc_max_disparity = better_disparity ? d : zncc_max_disparity;
        max_zncc = better_disparity ? zncc : max_zncc;
    }
    map[gy * width + gx] = zncc_max_disparity;
}

//Maps the disparity value scale to grayscale
inline uchar grayscale_disparity(
    const uchar min_disparity, const uchar max_disparity, const uchar value
){
    if(value <= min_disparity){
        return 0;
    }
    return 255.f * (value - min_disparity) / (max_disparity - min_disparity); 
}

//Calculate middle value using histogram
uchar calculate_window_non_zero_middle(int x, int y, int radius, __local uchar* pixels, const int width){
    int count = 0;
    int histogram[256] = {0};
    //Count the values with in the buckets
    for(int yr = -radius; yr <= radius; ++yr){
        __local uchar * pixel = &pixels[((y + yr) * width) + (x - radius)];
        for(int xr = -radius; xr <= radius; ++xr, ++pixel){
            if(*pixel > 0){
                histogram[*pixel]++;
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
    __global uchar* left, __global uchar* right, __global uchar* post_processed,
    int threshold_value, int window_radius, int min_disparity, int max_disparity, 
    const int width, const int height, __local uchar* left_tile, __local uchar* right_tile
){
    //pull locals
    const int local_width = get_local_size(0);
    const int local_height = get_local_size(1);

    //Calculate tile dimensions
    const int tile_height = (window_radius << 1) + local_height;
    const int tile_left_width = (window_radius << 1) + local_width;
    const int tile_right_width = local_width +  (window_radius << 1) + (max_disparity - min_disparity);

    const int group_base_x = local_width * get_group_id(0);
    const int group_base_y = local_height * get_group_id(1);
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    //Get tiled pixels
    for(int tile_y = ly; tile_y < tile_height; tile_y += local_height){
        int image_y = clamp(group_base_y + tile_y - window_radius, 0, height - 1);
        //Load left tile
        for(int tile_x = lx; tile_x < tile_left_width; tile_x += local_width){
            int image_x = clamp(group_base_x + tile_x - window_radius, 0, width - 1);
            left_tile[tile_y * tile_left_width + tile_x] = left[image_y * width + image_x];
        }
        //Load right tile
        for(int tile_x = lx; tile_x < tile_right_width; tile_x += local_width){
            int image_x = clamp(group_base_x + (tile_x - max_disparity - window_radius), 0, width - 1);
            right_tile[tile_y * tile_right_width + tile_x] = right[image_y * width + image_x];
        }
    }

    //Wait for whole group finish pulling pixel data
    barrier(CLK_LOCAL_MEM_FENCE);

    const int gx = get_global_id(0);
    const int gy = get_global_id(1);
    //Return on padded pixel
    if((gx >= width || gy >= height)){
        return;
    }

    //Tiling caused offset coordinates
    const int lx_halo = lx + window_radius;
    const int ly_halo = ly + window_radius;

    //Pull disparit L=>R and R=>L
    uchar disparity_value_l = left_tile[ly_halo * tile_left_width + lx_halo];
    uchar disparity_value_r = right_tile[ly_halo * tile_right_width + ((lx_halo + max_disparity) - disparity_value_l)];
    //Cross-check and occulsion
    uchar final_value = disparity_value_l;
    if(abs(disparity_value_l - disparity_value_r) > threshold_value || final_value == 0){
        //Get the window middle value as filler
        final_value = calculate_window_non_zero_middle(lx_halo, ly_halo, window_radius, left_tile, tile_left_width);
    }
    post_processed[gy * width + gx] = grayscale_disparity(min_disparity, max_disparity, final_value);
}