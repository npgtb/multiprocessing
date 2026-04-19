#ifndef MP_COURSE_IMAGE_H
#define MP_COURSE_IMAGE_H

#include <string>

namespace mp_course{
    enum class ImageFormat{
        UNKNOWN,
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
    };
}

#endif