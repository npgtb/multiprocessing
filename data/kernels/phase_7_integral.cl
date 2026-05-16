//Injected at compile time
#ifndef RADIUS 
#define RADIUS 1
#endif
#ifndef MIN_D
#define MIN_D 1
#endif
#ifndef MAX_D_EXC
#define MAX_D_EXC 1
#endif
#ifndef MAX_D_INC
#define MAX_D_INC 1
#endif
#ifndef THRESHOLD_VALUE
#define THRESHOLD_VALUE 1
#endif


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

//Integral image row pass 
__kernel void calculate_integral_map_rows(
    __global uchar* pixels, __global ulong* sum_map_data, __global ulong* square_map_data,
    const int img_w, const int img_h, const int map_w, const int right
){
    const int map_y = get_global_id(0);
    const int img_y = map_y - RADIUS;
    const int map_h = get_global_size(0);
    const int map_row = map_y * map_w;
    const int img_row = clamp(img_y, 0, img_h - 1) * img_w; 
    
    ulong row_sum = 0, row_square = 0;
    int x_start = right == 1 ? ((RADIUS*-1) - MAX_D_EXC) : -RADIUS;
    for(int img_x = x_start, map_x = 0; map_x < map_w; ++img_x, ++map_x){
        //For image clamp x => Replication
        const int img_i = img_row + clamp(img_x, 0, img_w - 1);
        const int map_i = map_row + map_x;
        //Get pixel value
        ulong value = pixels[img_i];
        row_sum += value;
        row_square += (value * value);
        //Store running values for the pixel
        sum_map_data[map_i] = row_sum;
        square_map_data[map_i] = row_square;
    }
}

//Integral image column pass
__kernel void calculate_integral_map_columns(
    __global ulong* sum_map_data, __global ulong* square_map_data,
    const int map_h
){
    const int map_x = get_global_id(0);
    const int map_w = get_global_size(0);
    ulong column_sum = 0, column_square = 0;
    for(int map_y = 0; map_y < map_h; ++map_y){
        const int map_i = (map_y * map_w) + map_x;
        //Get column values
        column_sum += sum_map_data[map_i];
        column_square += square_map_data[map_i];
        //Store running values for the pixel
        sum_map_data[map_i] = column_sum;
        square_map_data[map_i] = column_square;
    }
}

//Returns the map integral of the area as described
ulong integral_map_sum(
    __global ulong * map_pixels, const int map_w,
    const int center_x, const int center_y
){
    //Utilizing the formula described in
    //https://en.wikipedia.org/wiki/Summed-area_table
    const int x = max(center_x - RADIUS-1, 0);
    const int y = max(center_y - RADIUS-1, 0);
    const int x1 = center_x + RADIUS;
    const int y1 = center_y + RADIUS;

    ulong D = map_pixels[(y1 * map_w) + x1];
    ulong C = map_pixels[(y1 * map_w) + x];
    ulong B = map_pixels[(y * map_w) + x1];
    ulong A = map_pixels[(y * map_w) + x]; 
    return D + A - B - C; 
}

//Load local tiles from global memory for the left and right image
void load_tiles(
    const size_t lx, const size_t ly,
    const size_t lgw, const size_t lgh,
    const int d_dir, const int dr_pad,
    const int tlw, const int trw, const int iw, const int ih,
    local uchar* ltile, local uchar* rtile, 
    __global uchar* limage, __global uchar* rimage
){
    //Local tile height
    const int th = (RADIUS << 1) + lgh;
    //Local group base coordinates
    const int gbx = lgw * get_group_id(0);
    const int gby = lgh * get_group_id(1);

    //Get tiled pixels
    for(int tile_y = ly; tile_y < th; tile_y += lgh){      
        const int img_y_i = clamp(gby + tile_y - RADIUS, 0, ih - 1) * iw;
        const int tile_y_li = tile_y * tlw;
        const int tile_y_ri = tile_y * trw;
        //Load left tile
        for(int tile_x = lx; tile_x < tlw; tile_x += lgw){
            int image_x = clamp(gbx + tile_x - RADIUS, 0, iw - 1);
            ltile[tile_y_li + tile_x] = limage[img_y_i + image_x];
        }
        //Load right tile
        for(int tile_x = lx; tile_x < trw; tile_x += lgw){
            int shift_x = tile_x - RADIUS - dr_pad;
            int image_x = clamp(gbx + shift_x, 0, iw - 1);
            rtile[tile_y_ri + tile_x] = rimage[img_y_i + image_x];
        }
    }
    //Wait for whole group finish pulling pixel data
    barrier(CLK_LOCAL_MEM_FENCE);
}

//Calculates the disparity using the ZNCC algo. Calculates disparity shift from left image to right image, storing values in map image.
__kernel void calculate_disparity_map(
    __global uchar* limg, __global uchar* rimg, __global ulong* lsm, __global ulong * rsm,
     __global ulong* lsq, __global ulong* rsq, __global uchar* d_map,
    const int d_dir, const int imgw, const int imgh, const int integw, 
    const int integh, const int integ_lpad, const int integ_rpad,
    __local uchar * ltile, __local uchar * rtile
){
    //Local group coordinates and width and height
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int lgw = get_local_size(0);
    const int lgh = get_local_size(1);
    //Calculate tile widths
    const int tlw = (RADIUS << 1) + lgw;
    const int trw = lgw +  (RADIUS << 1) + (MAX_D_EXC);
    //Right image disparity padding
    const int dr_pad = (d_dir == -1 ? MAX_D_EXC : 0);

    //Load local tiles
    load_tiles(
        lx, ly, lgw, lgh, d_dir, 
        dr_pad, tlw, trw, imgw, imgh,
        ltile, rtile, limg, rimg
    );

    //Threads Global X,Y
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);
    //Dont calculate padded pixels
    if((gx >= imgw || gy >= imgh)){
        return;
    }

    //Tiling and integral caused offset coordinates
    const int integ_glx = gx + integ_lpad;
    const int integ_grx = gx + integ_rpad;
    const int integ_y = gy + RADIUS;
    const int llx_halo = lx + RADIUS;
    const int lrx_halo = llx_halo + dr_pad;
    const int ly_halo = ly + RADIUS;

    //Window pixel count and vector load end based on radius
    const int win_pix_count = ((RADIUS << 1) + 1) * ((RADIUS << 1) + 1);
    const int vload_size = 4;
    const int vload_end = (((RADIUS << 1) + 1) & ~(vload_size - 1));
    
    //Track zncc score
    float max_zncc = -1.f; 
    uchar zncc_MAX_D_EXC = 0;

    //Calculate Left mean and variance
    const float lmean = ((float) integral_map_sum(lsm, integw, integ_glx, integ_y)) / win_pix_count;
    const float lvar = ((float)integral_map_sum(lsq, integw, integ_glx, integ_y) / win_pix_count) - (lmean * lmean);
    for(uchar d = MIN_D; d <= MAX_D_EXC; ++d){
        //Shift tile and integral coords with disparity
        const int d_shift = (d * d_dir);
        int ldrx_halo = lrx_halo + d_shift;
        int integ_rdis = integ_grx + d_shift;

        //Calculate right mean and variance
        float rmean = ((float) integral_map_sum(rsm, integw, integ_rdis, integ_y)) / win_pix_count;
        float rvar = ((float)integral_map_sum(rsq, integw, integ_rdis, integ_y) / win_pix_count) - (rmean * rmean);

        //Calculate zncc score
        float acc_up = 0.f;
        float4 acc_vup = (float4)(0.0f);
        //Loop the window for upper
        for(int yr = -RADIUS; yr <= RADIUS; ++yr){
            int xr = -RADIUS;
            __local uchar * left_pixel = &ltile[((ly_halo + yr) * tlw) + (llx_halo - RADIUS)];
            __local uchar * right_pixel = &rtile[((ly_halo + yr) * trw) + (ldrx_halo - RADIUS)];
            //Vector load [vload_size] pixels 
            for(
                int i = 0; i < vload_end; 
                xr += vload_size, i += vload_size, 
                left_pixel += vload_size, right_pixel += vload_size
            ){
                //Adjust upper
                acc_vup += convert_float4(vload4(0, left_pixel)) * convert_float4(vload4(0, right_pixel));
            }
            //Handle the remainder
            for(; xr <= RADIUS; ++xr){
                //Adjust upper
                acc_up += (float)(*left_pixel++) * (float)(*right_pixel++);
            }
        }
        //Calculate zncc score
        acc_up += acc_vup.x + acc_vup.y + acc_vup.z + acc_vup.w - (win_pix_count * lmean * rmean);
        float result = acc_up / (win_pix_count * native_sqrt(lvar * rvar));
        result = select(0.f, result, isfinite(result));
        //Check if its new highscore
        bool better_disparity = result > max_zncc;
        zncc_MAX_D_EXC = better_disparity ? d : zncc_MAX_D_EXC;
        max_zncc = better_disparity ? result : max_zncc;
    }
    d_map[gy * imgw + gx] = zncc_MAX_D_EXC;
}

//Maps the disparity value scale to grayscale
inline uchar grayscale_disparity(const uchar value){
    //Normalize the disparity value, clamp it and return gs value
    float d_norm = (float)(value - MIN_D) / (float)(MAX_D_EXC - MIN_D);
    d_norm = clamp(d_norm, 0.0f, 1.0f);
    return (uchar)(255.f * d_norm);
}

//Calculate middle value using histogram
uchar calculate_window_non_zero_middle(int x, int y, __local uchar* pixels, const int width){
    int count = 0;
    ushort histogram[MAX_D_INC] = {0};
    //Count the values with in the buckets
    for(int yr = -RADIUS; yr <= RADIUS; ++yr){
        __local uchar * pixel = &pixels[((y + yr) * width) + (x - RADIUS)];
        for(int xr = -RADIUS; xr <= RADIUS; ++xr, ++pixel){
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
        ushort * histogram_value = &histogram[1];
        int sum = 0;
        for(int i = 1; i < MAX_D_INC; ++i){
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
    const int imgw, const int imgh, __local uchar* ltile, __local uchar* rtile
){
    //Local group coordinates and width and height
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int lgw = get_local_size(0);
    const int lgh = get_local_size(1);
    //Calculate tile widths
    const int tlw = (RADIUS << 1) + lgw;
    const int trw = lgw +  (RADIUS << 1) + (MAX_D_EXC);

    //Load local tiles
    load_tiles(
        lx, ly, lgw, lgh, -1, MAX_D_EXC, 
        tlw, trw, imgw, imgh,
        ltile, rtile, limg, rimg
    );

    //Threads Global X,Y
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);
    //Dont calculate padded pixels
    if((gx >= imgw || gy >= imgh)){
        return;
    }

    //Tiling caused offset coordinates
    const int lx_halo = lx + RADIUS;
    const int ly_halo = ly + RADIUS;

    //Pull disparity L=>R and R=>L
    uchar disparity_value_l = ltile[ly_halo * tlw + lx_halo];
    uchar disparity_value_r = rtile[ly_halo * trw + ((lx_halo + MAX_D_EXC) - disparity_value_l)];
    //Cross-check and occulsion
    uchar final_value = disparity_value_l;
    if(abs(disparity_value_l - disparity_value_r) > THRESHOLD_VALUE || final_value == 0){
        //Get the window middle value as filler
        final_value = calculate_window_non_zero_middle(lx_halo, ly_halo, ltile, tlw);
    }
    post_processed[gy * imgw + gx] = grayscale_disparity(final_value);
}