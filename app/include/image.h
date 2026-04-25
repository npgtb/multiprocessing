#ifndef MP_COURSE_IMAGE_H
#define MP_COURSE_IMAGE_H

#include <string>
#include <cstdint>

namespace mp{
    enum class ImageFormat{
        UNKNOWN,
        INTEGRAL, //uint32_t
        RGBA, //uint32_t
        GRAY //uint8_t
    };

    //C type image
    struct Image{
        void * pixels;
        int w, h;
        ImageFormat format;
        
        //Initialize empty image
        Image():pixels(nullptr), w(0), h(0), format(ImageFormat::UNKNOWN){}

        //RAII free memory
        ~Image(){
            free_memory();
        }

        //Free memory allocated
        void free_memory(){
            if(pixels){
                free(pixels);
                pixels = nullptr;
            }
        } 

        //Clamps the coordinate to the image 
        int clamp_x(int x){
            return std::min(std::max(x, 0), w-1);
        }

        //Clamps the coordinate to the image 
        int clamp_y(int y){
            return std::min(std::max(y, 0), h-1);
        }

        //Load image from a given path into pixels
        bool load_path(const std::string& path);

        //Save the image to the given path
        bool save(const std::string& path);

        //Sets the images data to the given parameters, freeing previous data
        template <typename T>
        requires (std::same_as<T, uint8_t> || std::same_as<T, uint32_t> || std::same_as<T, uint64_t>)
        void set(T*pixel_data, const int width, const int height, ImageFormat image_format){
            free_memory();
            pixels = static_cast<void*>(pixel_data);
            w = width;
            h = height;
            format = image_format;
        }
    };
}

#endif