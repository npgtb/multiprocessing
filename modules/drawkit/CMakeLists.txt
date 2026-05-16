add_library(DRAWKIT STATIC src/anchor.cc)

add_subdirectory(api)
add_subdirectory(drawkit_sdl3)

target_link_libraries(DRAWKIT
    PUBLIC DRAWKIT_API
    PRIVATE DRAWKIT_SDL3
)

if(ENABLE_ASANLEAK)
    message(STATUS "DRAWKIT Address sanitizer + leak enabled")
    target_compile_options(DRAWKIT PRIVATE
        -g
        -fsanitize=address
        -fsanitize=leak
        -fno-omit-frame-pointer
    )
    target_link_options(DRAWKIT PRIVATE
        -fsanitize=address
        -fsanitize=leak
    )
endif()