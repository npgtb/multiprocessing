#ifndef DRAWKIT_API_SURFACE_H
#define DRAWKIT_API_SURFACE_H

#include <memory>
#include <string>

#include <drawkit/pixel_format.h>

namespace drawkit{
    class Renderer;
    class Surface{
        public:
            struct SurfaceView{
                void* pixels; //Format of pixel structure depends on the format of the surface and pitch
                int width;
                int height;
                int pitch;
                PixelFormat format;
            };

            //Creates non initialized surface
            Surface();
            
            //Creates pixel data structure defined by the paramters
            bool create(const int width, const int height, const PixelFormat format);

            //Creates a surface from the given data
            bool create(const int width, const int height, int pitch, const PixelFormat format, void* pixels);

            //Tries to load the given path as a surface
            bool load(const std::string& path);

            //Tries to save the surface to the given file path
            bool save(const std::string& path);

            //Returns surtfaces width
            int width() const;

            //Return surfaces height
            int height() const;

            //Returns the distance in bytes between rows of pixels
            int pitch() const;

            //Returns the pixel format of the surface
            PixelFormat format() const;

            //Attempts to convert the pixel format of the surface
            bool convert_format(PixelFormat new_format);

            //Checks if the surface pixel data is locked
            bool is_locked() const;

            //Checks wheter we should use lock, before accesing surface data
            bool should_lock() const;

            //Tries to lock the surface pixel data
            bool lock();

            //Unlocks the surfaces pixel data
            void unlock();

            //Enables viewing of the pixel data
            SurfaceView view();

            //RAII
            ~Surface();

            //Destroyes the surface object
            void close();

        private:
        
            bool locked_;
            struct Implementation;
            std::unique_ptr<Implementation> implementation_;
    };
}
#endif