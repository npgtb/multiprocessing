add_library(DRAWKIT STATIC src/anchor.cc)

add_subdirectory(api)
add_subdirectory(drawkit_sdl3)

target_link_libraries(DRAWKIT
    PUBLIC DRAWKIT_API
    PRIVATE DRAWKIT_SDL3
)