#ifndef MP_COURSE_IMAGE_H
#define MP_COURSE_IMAGE_H

#include <string>
#include <vector>
#include <cstdint>
#include <variant>

namespace mp{
    using PixelData = std::variant<
        std::monostate,
        std::vector<uint8_t>,
        std::vector<uint32_t>,
        std::vector<uint64_t>
    >;
    
    enum class ImageFormat{
        UNKNOWN,
        INTEGRAL, //uint64_t
        RGBA, //uint32_t
        GRAY //uint8_t
    };

    //C type image
    struct Image{
        PixelData pixels;
        int w, h;
        ImageFormat format;
        int x_pad, y_pad;
        
        //Initialize empty image
        Image():w(0), h(0), format(ImageFormat::UNKNOWN), x_pad(0), y_pad(0){}

        //RAII free memory
        ~Image(){}

        //Clamps the coordinate to the image 
        inline int clamp_x(int x){
            return std::min(std::max(x, 0), w-1);
        }

        //Clamps the coordinate to the image 
        inline int clamp_y(int y){
            return std::min(std::max(y, 0), h-1);
        }

        //Applies padding to the image coordinate x component
        inline int pad_x(const int x){
            return x + x_pad;
        }

        //Applies padding to the image coordinate y component
        inline int pad_y(const int y){
            return y + y_pad;
        }

        //Load image from a given path into pixels
        bool load_path(const std::string& path);

        //Save the image to the given path
        bool save(const std::string& path);

        //Get a pointer to the internal vector
        template <typename T>
        requires (std::same_as<T, uint8_t> || std::same_as<T, uint32_t> || std::same_as<T, uint64_t>)
        std::vector<T>* storage(){
            return std::get_if<std::vector<T>>(&pixels);
        }

        //Get a data pointer to the internal c array
        template <typename T>
        requires (std::same_as<T, uint8_t> || std::same_as<T, uint32_t> || std::same_as<T, uint64_t>)
        T* data(){
            if (auto* vec = std::get_if<std::vector<T>>(&pixels)) {
                return vec->data();
            }
            throw std::bad_variant_access();
        }

        //Initialize the image to the memory type
        template <typename T>
        requires (std::same_as<T, uint8_t> || std::same_as<T, uint32_t> || std::same_as<T, uint64_t>)
        void init(const int width, const int height, ImageFormat image_format){
            //Set the storage
            pixels.emplace<std::vector<T>>(width * height);
            w = width;
            h = height;
            format = image_format;
        }

        //Initialize the image to the memory type
        template <typename T>
        requires (std::same_as<T, uint8_t> || std::same_as<T, uint32_t> || std::same_as<T, uint64_t>)
        void init(const int width, const int height, const int padding_x, const int padding_y, ImageFormat image_format){
            //Set the storage
            pixels.emplace<std::vector<T>>(width * height);
            w = width;
            h = height;
            x_pad = padding_x;
            y_pad = padding_y;
            format = image_format;
        }

        //Set the image to the given data
        template <typename T>
        requires (std::same_as<T, uint8_t> || std::same_as<T, uint32_t> || std::same_as<T, uint64_t>)
        void set(std::vector<T>&& data, const int width, const int height, ImageFormat image_format){
            //Set the storage
            pixels = data;
            w = width;
            h = height;
            format = image_format;
        }

        //Set the image to the given data
        template <typename T>
        requires (std::same_as<T, uint8_t> || std::same_as<T, uint32_t> || std::same_as<T, uint64_t>)
        void set(std::vector<T>&& data, const int width, const int height, const int padding_x, const int padding_y, ImageFormat image_format){
            //Set the storage
            pixels = data;
            w = width;
            h = height;
            x_pad = padding_x;
            y_pad = padding_y;
            format = image_format;
        }
    };
}

#endif