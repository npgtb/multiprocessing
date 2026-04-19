#include <image.h>
#include <cstring>
#include <drawkit/surface.h>

namespace mp_course{

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

        //Save the image to the given path
        bool Image::save(const std::string& path){
            //Assume RGBA image
            drawkit::PixelFormat surface_format = drawkit::PixelFormat::RGBA8;
            int pitch = w * sizeof(uint32_t);
            if(format == ImageFormat::GRAY){
                //Adjust for grayscale
                surface_format = drawkit::PixelFormat::GRAY8;
                pitch = w;
            }
            drawkit::Surface surface;
            //Try to convert pixel data into surface
            if(surface.create(w, h, pitch, surface_format, pixels)){
                return surface.save(path);
            }
            return false;
        }
}