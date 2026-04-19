#include "drawkit/pixel_format.h"
#include <SDL3/SDL_pixels.h>
#include <SDL3/SDL_surface.h>
#include <SDL3_image/SDL_image.h>

#include <drawkit/surface.h>
#include <drawkit_sdl3/surface.h>
#include <drawkit_sdl3/conversion.h>

namespace drawkit{
    //Creates non initialized surface
    Surface::Surface():locked_(false), implementation_(std::make_unique<Surface::Implementation>()){}
    
    //Creates pixel data structure defined by the paramters
    bool Surface::create(const int width, const int height, const PixelFormat format){
        close();
        namespace sdl_conversion = drawkit::conversion::to_sdl;
        implementation_->surface = SDL_CreateSurface(width, height, sdl_conversion::make_pixel_format(format));
        if(implementation_->surface && format == PixelFormat::GRAY8){
            //SDL needs palette for grayscale images.
            return grayscale_palette(implementation_->surface);
        }
        return !(implementation_->surface == nullptr);
    }

    //Creates a surface from the given data
    bool Surface::create(const int width, const int height, int pitch, const PixelFormat format, void* pixels){
        close();
        namespace sdl_conversion = drawkit::conversion::to_sdl;
        implementation_->surface = SDL_CreateSurfaceFrom(width, height, sdl_conversion::make_pixel_format(format), pixels, pitch);
        if(implementation_->surface && format == PixelFormat::GRAY8){
            //SDL needs palette for grayscale images.
            return grayscale_palette(implementation_->surface);
        }
        return !(implementation_->surface == nullptr);
    }

    //Tries to load the given path as a surface
    bool Surface::load(const std::string& path){
        close();
        implementation_->surface =  IMG_Load(path.c_str());
        //IMG_Load returns NULL when it fails
        return !(implementation_->surface == nullptr);
    }

    //Tries to save the surface to the given file path
    bool Surface::save(const std::string& path){
        if(implementation_->surface){
            return IMG_Save(implementation_->surface, path.c_str());
        }
        return false;
    }

    //Returns surtfaces width
    int Surface::width() const{
        if(implementation_->surface){
            return implementation_->surface->w;
        }
        return 0;
    }

    //Return surfaces height
    int Surface::height() const{
        if(implementation_->surface){
            return implementation_->surface->h;
        }
        return 0;
    }

    //Returns the distance in bytes between rows of pixels
    int Surface::pitch() const{
        if(implementation_->surface){
            return implementation_->surface->pitch;
        }
        return 0;
    }

    //Returns the pixel format of the surface
    PixelFormat Surface::format() const{
        if(implementation_->surface){
            namespace drawkit_conversion = drawkit::conversion::to_drawkit;
            return drawkit_conversion::make_pixel_format(implementation_->surface->format);
        }
        return PixelFormat::UNKNOWN;
    }

    //Attempts to convert the pixel format of the surface
    bool Surface::convert_format(PixelFormat new_format){
        if(implementation_->surface){
            namespace sdl_conversion = drawkit::conversion::to_sdl;
            SDL_Surface * formated_surface = SDL_ConvertSurface(
                implementation_->surface,
                 sdl_conversion::make_pixel_format(new_format)
            );
            if(formated_surface){
                close();
                implementation_->surface = formated_surface;
                return true;
            }
        }
        return false;
    }

    //Checks if the surface pixel data is locked
    bool Surface::is_locked() const{
        return locked_;
    }

    //Checks wheter we should use lock, before accesing surface data
    bool Surface::should_lock() const{
        if(implementation_->surface){
            return SDL_MUSTLOCK(implementation_->surface);
        }
        return false;
    }

    //Tries to lock the surface pixel data
    bool Surface::lock(){
        if(implementation_->surface){
            locked_ = SDL_LockSurface(implementation_->surface);
            return locked_;
        }
        return false;
    }

    //Unlocks the surfaces pixel data
    void Surface::unlock(){
        if(implementation_->surface && locked_){
            SDL_UnlockSurface(implementation_->surface);
            locked_ = false;
        }
    }

    //Enables viewing of the pixel data
    Surface::SurfaceView Surface::view(){
        if(implementation_->surface){
            namespace drawkit_conversion = drawkit::conversion::to_drawkit;
            return {
                implementation_->surface->pixels,
                implementation_->surface->w,
                implementation_->surface->h,
                implementation_->surface->pitch,
                drawkit_conversion::make_pixel_format(implementation_->surface->format)
            };
        }
        return {nullptr, 0, 0, 0, PixelFormat::UNKNOWN};
    }

    //RAII
    Surface::~Surface(){
        close();
    }

    //Destroyes the surface object
    void Surface::close(){
        if(implementation_->surface){
            SDL_DestroySurface(implementation_->surface);
            implementation_->surface = nullptr;
        }
    }

    //Appends a grayscale palette to the surface
    bool grayscale_palette(SDL_Surface * surface){
        //SDL Uses indexing in INDEX8 format to determine the color
        //From the palette. palette[VALUE] => color.
        constexpr int palette_colors = 256;
        SDL_Palette * grayscale_palette = SDL_CreatePalette(palette_colors);
        bool successful = false;
        if(grayscale_palette){
            SDL_Color grayscale_colors_colors[palette_colors];
            for (int i = 0; i < palette_colors; ++i){
                grayscale_colors_colors[i].r = i;
                grayscale_colors_colors[i].g = i;
                grayscale_colors_colors[i].b = i;
                grayscale_colors_colors[i].a = 255;
            }
            if(SDL_SetPaletteColors(grayscale_palette, grayscale_colors_colors, 0, palette_colors)){
                successful = SDL_SetSurfacePalette(surface, grayscale_palette);
            }
            SDL_DestroyPalette(grayscale_palette);
        }
        return successful;
    }
}