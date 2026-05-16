#include <image.h>
#include <cstring>
#include <drawkit/surface.h>

namespace mp{

    //Transfers the pixels from surface to given storage
    bool transfer_pixel_data_from_surface(drawkit::Surface& surface, Image& image){
        //Memory for RGBA/ABGR image
        image.init<uint32_t>(surface.width(), surface.height(), ImageFormat::RGBA);
        uint32_t* target_storage = image.data<uint32_t>();
        if(target_storage && (!surface.should_lock() || (surface.should_lock() && surface.lock()))){
            drawkit::Surface::SurfaceView view = surface.view();
            uint8_t * source_bytes = (uint8_t*)view.pixels;
            uint8_t * source_row = nullptr;
            //Plan:memcpy row by row from surface, while accounting for potential pitch
            for(int i = 0; i < surface.height(); ++i){
                //Calculate row starting point in memory
                source_row = (uint8_t*)(source_bytes + (i * view.pitch));                 
                //Copy it to the continious array, copy width * 4 bytes
                memcpy(&target_storage[view.width * i], source_row, (view.width * sizeof(uint32_t)));
            }
            surface.unlock();
            return true;
        }
        return false;
    }

    //Load image from a given path into pixels
    bool Image::load_path(const std::string& path){
        drawkit::Surface surface;
        if(surface.load(path)){
            if(
                //We are either in 32-bit format (LE/BE)
                (surface.format() == drawkit::PixelFormat::RGBA8 && surface.format() == drawkit::PixelFormat::ABGR8) ||
                //Or we manage to convert into one
                surface.convert_format(drawkit::PixelFormat::RGBA8)
            )
            {
                return transfer_pixel_data_from_surface(surface, *this);
            }
        }
        return false;
    }

    //Produce a grayscale image by normalizing the data
    template <typename T>
    requires (std::same_as<T, uint8_t> || std::same_as<T, uint32_t> || std::same_as<T, uint64_t>)
    std::vector<uint8_t> normalized_grayscale(T * data, int w, int h){
        T largest = 0;
        int pixel_count = w * h;
        for(int i = 0; i < pixel_count; ++i){
            if(largest < data[i]){
                largest = data[i];
            }
        }
        std::vector<uint8_t> normalized_data(pixel_count);
        if(largest > 0){
            float inv_max = 1.f / largest;
            for(int i = 0; i < pixel_count; ++i){
                float normalized = ((float)data[i] * inv_max);
                normalized_data[i] = static_cast<uint8_t>(normalized * 255.f + 0.5f);
            }
        }
        return std::move(normalized_data);
    }

    //Save the image to the given path
    bool Image::save(const std::string& path){
        bool free_normalized = false;
        drawkit::PixelFormat surface_format = drawkit::PixelFormat::UNKNOWN;
        std::vector<uint8_t> normalized_data;
        drawkit::Surface surface;
        switch(format){
            case mp::ImageFormat::RGBA:
                if(surface.create(w, h, (w * sizeof(uint32_t)), drawkit::PixelFormat::RGBA8, this->data<uint32_t>())){
                    return surface.save(path);
                }
            break;
            case mp::ImageFormat::GRAY:
                if(surface.create(w, h, w, drawkit::PixelFormat::GRAY8, this->data<uint8_t>())){
                    return surface.save(path);
                }
            break;
            case mp::ImageFormat::INTEGRAL:
                normalized_data = normalized_grayscale(this->data<uint64_t>(), w, h);
                if(surface.create(w, h, w, drawkit::PixelFormat::GRAY8, normalized_data.data())){
                    return surface.save(path);
                }
            break;
            default:
                return false;
        }
        return false;
    }
}