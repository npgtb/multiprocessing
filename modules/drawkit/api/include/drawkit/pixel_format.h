#ifndef DRAWKIT_API_PIXEL_FORMAT_H
#define DRAWKIT_API_PIXEL_FORMAT_H

namespace drawkit{
    enum class PixelFormat{
        UNKNOWN,
        //32
        RGBA8, //Big endian
        ABGR8, //Little endian
        BGRA8, //Big endian
        ARGB8, //Little endian

        //24
        RGB8,
        BGR8,
        //16
        RGB565,
        RGBA5551,
        //8
        GRAY8,
    };
}

#endif