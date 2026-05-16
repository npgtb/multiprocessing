#ifndef DRAWKIT_SDL3_SURFACE_H
#define DRAWKIT_SDL3_SURFACE_H

#include <SDL3/SDL_surface.h>

#include <drawkit/surface.h>

namespace drawkit{
    struct Surface::Implementation{
        SDL_Surface * surface = nullptr;
    };

    //Appends a grayscale palette to the surface
    bool grayscale_palette(SDL_Surface * surface);
}

#endif