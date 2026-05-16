#ifndef DRAWKIT_SDL3_CONVERSION_H
#define DRAWKIT_SDL3_CONVERSION_H

#include <SDL3/SDL.h>
#include <drawkit/pixel_format.h>

namespace drawkit::conversion{
    namespace to_sdl{

        //Transforms drawkit::PixelFormat to SDL_PixelFormat
        SDL_PixelFormat make_pixel_format(const drawkit::PixelFormat original);
    }

    namespace to_drawkit{

        //Transforms SDL_PixelFormat to drawkit::PixelFormat
        drawkit::PixelFormat make_pixel_format(const SDL_PixelFormat original);
    }
}

#endif