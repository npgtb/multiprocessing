#include <image.h>
#include <cstring>
#include <drawkit/surface.h>

namespace mp{

        //Transfers the pixels from surface to given storage
        bool transfer_pixel_data_from_surface(drawkit::Surface& surface, void** storage){
            //32-bit format (LE/BE)
            if((surface.format() == drawkit::PixelFormat::RGBA8 || surface.format() == drawkit::PixelFormat::ABGR8)){
                //Memory for RGBA/ABGR image
                *storage = malloc(surface.width() * surface.height() * sizeof(uint32_t));
                if(*storage && (!surface.should_lock() || (surface.should_lock() && surface.lock()))){
                    drawkit::Surface::SurfaceView view = surface.view();
                    uint8_t * source_bytes = (uint8_t*)view.pixels;
                    uint8_t * source_row = nullptr;
                    uint32_t * target_storage = (uint32_t*)*storage;
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
                //Clean up the failed attempt
                if(*storage){
                    free(*storage);
                    *storage = nullptr;
                }
            }
            return false;
        }

        //Load image from a given path into pixels
        bool Image::load_path(const std::string& path){
            free_memory();
            drawkit::Surface surface;
            if(surface.load(path)){
                if(
                    //We are either in 32-bit format (LE/BE)
                    (surface.format() == drawkit::PixelFormat::RGBA8 && surface.format() == drawkit::PixelFormat::ABGR8) ||
                    //Or we manage to convert into one
                    surface.convert_format(drawkit::PixelFormat::RGBA8)
                )
                {
                    if(transfer_pixel_data_from_surface(surface, &pixels)){
                        format = ImageFormat::RGBA;
                        w = surface.width();
                        h = surface.height();
                        return true;
                    }
                }
            }
            return false;
        }

        //Produce a grayscale image by normalizing the data
        template <typename T>
        requires (std::same_as<T, uint8_t> || std::same_as<T, uint32_t> || std::same_as<T, uint64_t>)
        uint8_t * normalized_grayscale(T * data, int w, int h){
            int pixel_count = w * h;
            T largest = 0;
            for(int i = 0; i < pixel_count; ++i){
                if(largest < data[i]){
                    largest = data[i];
                }
            }
            uint8_t * normalized_data = static_cast<uint8_t*>(malloc(pixel_count));
            if(normalized_data){
                for(int i = 0; i < pixel_count; ++i){
                    float normalized = ((float)data[i] / largest);
                    normalized_data[i] = (uint8_t)(255.f * normalized);
                }
                return normalized_data;
            }
            return nullptr;
        }

        //Save the image to the given path
        bool Image::save(const std::string& path){
            bool free_normalized = false;
            drawkit::PixelFormat surface_format = drawkit::PixelFormat::UNKNOWN;
            void * data = nullptr;
            int pitch = 0;

            switch(format){
                case mp::ImageFormat::RGBA:
                    data = pixels;
                    pitch = w * sizeof(uint32_t);
                    surface_format = drawkit::PixelFormat::RGBA8;
                break;
                case mp::ImageFormat::GRAY:
                    data = pixels;
                    pitch = w;
                    surface_format = drawkit::PixelFormat::GRAY8;
                break;
                case mp::ImageFormat::INTEGRAL:
                    data = normalized_grayscale(static_cast<uint32_t*>(pixels), w, h);
                    pitch = w;
                    surface_format = drawkit::PixelFormat::GRAY8;
                    free_normalized = true;
                break;
                default:
                    return false;
            }

            bool success = false;
            drawkit::Surface surface;
            //Try to convert pixel data into surface
            if((success = surface.create(w, h, pitch, surface_format, data)) == true){
                success = surface.save(path);
            }

            if(free_normalized && data){
                free(data);
            }
            return success;
        }
}