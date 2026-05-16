add_library(DRAWKIT_API INTERFACE)
target_include_directories(DRAWKIT_API
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(ENABLE_ASANLEAK)
    message(STATUS "DRAWKIT_API Address sanitizer + leak enabled")
    target_compile_options(DRAWKIT_API INTERFACE
        -g
        -fsanitize=address
        -fsanitize=leak
        -fno-omit-frame-pointer
    )
    target_link_options(DRAWKIT_API INTERFACE
        -fsanitize=address
        -fsanitize=leak
    )
endif()