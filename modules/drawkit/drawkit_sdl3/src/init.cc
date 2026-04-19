#include <SDL3/SDL_init.h>

#include <drawkit/init.h>

namespace drawkit{
    //Initializes any libraries needed for functionality
    bool init(){
        if(SDL_Init(SDL_INIT_VIDEO)){
            return true;
        }
        return false;
    }

    //Shuts down any libraries needed for functianality.
    void shutdown(){
        SDL_Quit();
    }
};
