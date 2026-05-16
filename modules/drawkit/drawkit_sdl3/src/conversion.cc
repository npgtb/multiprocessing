
#include "drawkit/pixel_format.h"
#include <SDL3/SDL_pixels.h>
#include <drawkit_sdl3/conversion.h>

namespace drawkit::conversion{

    namespace to_sdl{
        //Transforms drawkit::PixelFormat to SDL_PixelFormat
        SDL_PixelFormat make_pixel_format(const drawkit::PixelFormat original){
            switch (original) {
                case drawkit::PixelFormat::UNKNOWN:
                    return SDL_PixelFormat::SDL_PIXELFORMAT_UNKNOWN;
                case drawkit::PixelFormat::RGBA8:
                    return SDL_PixelFormat::SDL_PIXELFORMAT_RGBA32;
                case drawkit::PixelFormat::ABGR8:
                    return SDL_PixelFormat::SDL_PIXELFORMAT_ABGR32;
                case drawkit::PixelFormat::BGRA8:
                    return SDL_PixelFormat::SDL_PIXELFORMAT_BGRA32;
                case drawkit::PixelFormat::ARGB8:
                    return SDL_PixelFormat::SDL_PIXELFORMAT_ARGB32;
                case drawkit::PixelFormat::RGB8:
                    return SDL_PixelFormat::SDL_PIXELFORMAT_RGB24;
                case drawkit::PixelFormat::BGR8:
                    return SDL_PixelFormat::SDL_PIXELFORMAT_BGR24;
                case drawkit::PixelFormat::RGB565:
                    return SDL_PixelFormat::SDL_PIXELFORMAT_RGB565;
                case drawkit::PixelFormat::RGBA5551:
                    return SDL_PixelFormat::SDL_PIXELFORMAT_RGBA5551;
                case drawkit::PixelFormat::GRAY8:
                    return SDL_PixelFormat::SDL_PIXELFORMAT_INDEX8;
                default:
                    return SDL_PixelFormat::SDL_PIXELFORMAT_UNKNOWN;
            }
        }
    }

    namespace to_drawkit{
        //Transforms SDL_PixelFormat to drawkit::PixelFormat
        drawkit::PixelFormat make_pixel_format(const SDL_PixelFormat original){
            switch (original) {
                case SDL_PixelFormat::SDL_PIXELFORMAT_UNKNOWN:
                    return drawkit::PixelFormat::UNKNOWN;
                case SDL_PixelFormat::SDL_PIXELFORMAT_RGBA32:
                    return drawkit::PixelFormat::RGBA8;
                case SDL_PixelFormat::SDL_PIXELFORMAT_ABGR32:
                    return drawkit::PixelFormat::ABGR8;
                case SDL_PixelFormat::SDL_PIXELFORMAT_BGRA32:
                    return drawkit::PixelFormat::BGRA8;
                case SDL_PixelFormat::SDL_PIXELFORMAT_ARGB32:
                    return drawkit::PixelFormat::ARGB8;
                case SDL_PixelFormat::SDL_PIXELFORMAT_RGB24:
                    return drawkit::PixelFormat::RGB8;
                case SDL_PixelFormat::SDL_PIXELFORMAT_BGR24:
                    return drawkit::PixelFormat::BGR8;
                case SDL_PixelFormat::SDL_PIXELFORMAT_RGB565:
                    return drawkit::PixelFormat::RGB565;
                case SDL_PixelFormat::SDL_PIXELFORMAT_RGBA5551:
                    return drawkit::PixelFormat::RGBA5551;
                case SDL_PixelFormat::SDL_PIXELFORMAT_INDEX8:
                    return drawkit::PixelFormat::GRAY8;
                default:
                    return drawkit::PixelFormat::UNKNOWN;
            }
        }
    }

}