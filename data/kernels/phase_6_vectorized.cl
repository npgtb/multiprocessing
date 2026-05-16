
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
    grayscaled[y * width + x] = (original_pixel.w * 218 + original_pixel.z * 732 + original_pixel.y * 74) >> 10;
}

//Calculates mean from the window in the given image. Expects grayscale image
inline float calculate_window_mean(
    const int x, const int y, const int radius, 
    __local uchar* tile, const int tw, const int win_pix_count,
    const int vload_end, const int vload_size
){
    float sum = 0;
    //Go trough the window
    for(int yr = -radius; yr <= radius; ++yr){
        int xr = -radius;
        __local uchar * pixel = &tile[((y + yr) * tw) + (x - radius)];
        //Handle [vload_size] pixels at time
        for(int i = 0; i < vload_end; xr += vload_size, i += vload_size, pixel += vload_size){
            uchar4 pixel_vector = vload4(0, pixel);
            sum += pixel_vector.x + pixel_vector.y + pixel_vector.z + pixel_vector.w;
        }
        //Handle the remainder
        for(; xr <= radius; ++xr){
            sum += *pixel++;
        }
    }
    return sum / win_pix_count;
}

//Calculates the ZNCC trough upper and lower sums. Expects grayscaled images
inline float calculate_zncc(
    int lx, int lxr, int ly, int radius, float lmean, float rmean, 
    __local uchar* ltile, __local uchar* rtile, const int tlw, const int trw,
    const int vload_end, const int vload_size
){
    float upper = 0.f;
    float lower_l = 0.f, lower_r = 0.f;

    float4 lmean4 = (float4)lmean;
    float4 rmean4 = (float4)rmean;
    for(int yr = -radius; yr <= radius; ++yr){
        __local uchar * left_pixel = &ltile[((ly + yr) * tlw) + (lx - radius)];
        __local uchar * right_pixel = &rtile[((ly + yr) * trw) + (lxr - radius)];
        int xr = -radius;
        //Handle 4 pixels at one time
        for(int i = 0; i < vload_end; xr += vload_size, i += vload_size, left_pixel += vload_size, right_pixel += vload_size){
            //Normalize brightness with means
            float4 left_pixel_vector = convert_float4(vload4(0, left_pixel)) - lmean4;
            float4 right_pixel_vector = convert_float4(vload4(0, right_pixel)) - rmean4;
            //Adjust upper and lower sums
            upper += dot(left_pixel_vector, right_pixel_vector);
            lower_l += dot(left_pixel_vector, left_pixel_vector);
            lower_r += dot(right_pixel_vector, right_pixel_vector);
        }
        //Handle the remainder
        for(; xr <= radius; ++xr){
            //Normalize brightness with means
            float l_diff = (float)(*left_pixel++) - lmean;
            float r_diff = (float)(*right_pixel++) - rmean;
            //Adjust upper and lower sums
            upper += l_diff * r_diff;
            lower_l += (l_diff * l_diff);
            lower_r += (r_diff * r_diff);
        }
    }
    float result = upper / (native_sqrt(lower_l * lower_r));
    return select(0.f, result, isfinite(result));
}

//Load local tiles from global memory for the left and right image
void load_tiles(
    const size_t lx, const size_t ly,
    const size_t lgw, const size_t lgh, const int radius,
    const int max_d, const int d_dir, const int dr_pad,
    const int tlw, const int trw, const int iw, const int ih,
    local uchar* ltile, local uchar* rtile, 
    __global uchar* limage, __global uchar* rimage
){
    //Local tile height
    const int th = (radius << 1) + lgh;
    //Local group base coordinates
    const int gbx = lgw * get_group_id(0);
    const int gby = lgh * get_group_id(1);

    //Get tiled pixels
    for(int tile_y = ly; tile_y < th; tile_y += lgh){      
        const int img_y_i = clamp(gby + tile_y - radius, 0, ih - 1) * iw;
        const int tile_y_li = tile_y * tlw;
        const int tile_y_ri = tile_y * trw;
        //Load left tile
        for(int tile_x = lx; tile_x < tlw; tile_x += lgw){
            int image_x = clamp(gbx + tile_x - radius, 0, iw - 1);
            ltile[tile_y_li + tile_x] = limage[img_y_i + image_x];
        }
        //Load right tile
        for(int tile_x = lx; tile_x < trw; tile_x += lgw){
            int shift_x = tile_x - radius - dr_pad;
            int image_x = clamp(gbx + shift_x, 0, iw - 1);
            rtile[tile_y_ri + tile_x] = rimage[img_y_i + image_x];
        }
    }
    //Wait for whole group finish pulling pixel data
    barrier(CLK_LOCAL_MEM_FENCE);
}

//Calculates the disparity using the ZNCC algo. Calculates disparity shift from left image to right image, storing values in map image.
__kernel void calculate_disparity_map(
    __global uchar* limg, __global uchar* rimg, __global uchar* d_map,
    const int radius, const int min_d, const int max_d, const int d_dir,
    const int imgw, const int imgh, __local uchar * ltile, __local uchar * rtile
){
    //Local group coordinates and width and height
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int lgw = get_local_size(0);
    const int lgh = get_local_size(1);
    //Calculate tile widths
    const int tlw = (radius << 1) + lgw;
    const int trw = lgw +  (radius << 1) + (max_d);
    //Right image disparity padding
    const int dr_pad = (d_dir == -1 ? max_d : 0);

    //Load local tiles
    load_tiles(
        lx, ly, lgw, lgh, radius, max_d, 
        d_dir, dr_pad, tlw, trw, imgw, imgh,
        ltile, rtile, limg, rimg
    );

    const int gx = get_global_id(0);
    const int gy = get_global_id(1);
    //Dont calculate padded pixels
    if((gx >= imgw || gy >= imgh)){
        return;
    }
    //Tiling caused offset coordinates
    const int llx_halo = lx + radius;
    const int lrx_halo = llx_halo + dr_pad;
    const int ly_halo = ly + radius;

    //Vectorized load variables
    const int vload_size = 4;
    const int vload_end = (((radius << 1) + 1) & ~(vload_size - 1));
    //Pixel count in a window
    const int win_pix_count = (((radius << 1) + 1) * ((radius << 1) + 1));

    float max_zncc = -1.f; 
    uchar zncc_max_d = 0;
    //Calculate Left mean
    float left_mean = calculate_window_mean(llx_halo, ly_halo, radius, ltile, tlw, win_pix_count, vload_end, vload_size);
    for(uchar d = min_d; d <= max_d; ++d){
        int ldrx_halo = lrx_halo + (d * d_dir);
        //Calculate right mean and zncc score
        float right_mean = calculate_window_mean(ldrx_halo, ly_halo, radius, rtile, trw, win_pix_count, vload_end, vload_size);
        float zncc = calculate_zncc(
            llx_halo, ldrx_halo, ly_halo, radius, 
            left_mean, right_mean, ltile, rtile, tlw, trw,
            vload_end, vload_size
        );
        //Check if we have new highscore
        bool better_disparity = zncc > max_zncc;
        zncc_max_d = better_disparity ? d : zncc_max_d;
        max_zncc = better_disparity ? zncc : max_zncc;
    }
    d_map[gy * imgw + gx] = zncc_max_d;
}

//Maps the disparity value scale to grayscale
inline uchar grayscale_disparity(
    const uchar min_d, const uchar max_d, const uchar value
){
    if(value <= min_d){
        return 0;
    }
    return 255.f * (value - min_d) / (max_d - min_d); 
}

//Calculate middle value using histogram
inline uchar calculate_window_sampled_middle(int x, int y, int radius, __local uchar* pixels, const int width){
    //Sample max 25 values
    uchar samples[25] = {0};
    int current_sample = 0;
    int sampling_step = max(1, radius/2);
    //Sample non zero values from the window
    for(int yr = -radius; yr <= radius; yr += sampling_step){
        int cy = (y + yr) * width;
        for(int xr = -radius; xr <= radius; xr += sampling_step){
            uchar pixel_value = pixels[cy + (x + xr)];
            if(pixel_value > 0){
                samples[current_sample] = pixel_value;
                ++current_sample;
            }
        }
    }
    //Do we have samples?
    if(current_sample != 0){
        int middle_sample = current_sample >> 1;
        for(int i = 0; i <= middle_sample; ++i){
            //Sort half the array from min > max
            int min_index = i;
            //Find the smallest
            for(int j = i +1; j < current_sample; ++j){
                if(samples[j] < samples[min_index]){
                    min_index = j;
                }
            }
            //Swap
            uchar temp = samples[i];
            samples[i] = samples[min_index];
            samples[min_index] = temp;
        }
        //Return middle value
        return samples[middle_sample];
    }
    return 0;
}

//Cross-checks the two disparity maps against each other
__kernel void cross_check_occulsion_disparity_maps(
    __global uchar* limg, __global uchar* rimg, __global uchar* post_processed,
    int threshold_value, int radius, int min_d, int max_d, 
    const int imgw, const int imgh, __local uchar* ltile, __local uchar* rtile
){
    //Local group coordinates and width and height
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int lgw = get_local_size(0);
    const int lgh = get_local_size(1);
    //Calculate tile widths
    const int tlw = (radius << 1) + lgw;
    const int trw = lgw +  (radius << 1) + (max_d);

    //Load local tiles
    load_tiles(
        lx, ly, lgw, lgh, radius, max_d, 
        -1, max_d, tlw, trw, imgw, imgh,
        ltile, rtile, limg, rimg
    );

    const int gx = get_global_id(0);
    const int gy = get_global_id(1);
    //Return on padded pixel
    if((gx >= imgw || gy >= imgh)){
        return;
    }

    const int lx_halo = lx + radius;
    const int ly_halo = ly + radius;
    //Get disparity values L=>R and R=>L
    uchar disparity_value_l = ltile[ly_halo * tlw + lx_halo];
    uchar disparity_value_r = rtile[ly_halo * trw + ((lx_halo + max_d) - disparity_value_l)];
    //Cross-check and occulsion
    uchar final_value = disparity_value_l;
    if(abs(disparity_value_l - disparity_value_r) > threshold_value || final_value == 0){
        //Get the window middle value as filler
        final_value = calculate_window_sampled_middle(lx_halo, ly_halo, radius, ltile, tlw);
    }
    post_processed[gy * imgw + gx] = grayscale_disparity(min_d, max_d, final_value);
}