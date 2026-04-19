find_package(SDL3 CONFIG REQUIRED)
find_package(SDL3_image CONFIG REQUIRED)

SET(DRAWKIT_SDL3_SRC
    src/init.cc
    src/conversion.cc
    src/high_performance_timer.cc
    src/surface.cc
)

add_library(DRAWKIT_SDL3 STATIC ${DRAWKIT_SDL3_SRC})
target_include_directories(DRAWKIT_SDL3
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(DRAWKIT_SDL3 PRIVATE 
    DRAWKIT_API
    SDL3::SDL3
    $<IF:$<TARGET_EXISTS:SDL3_image::SDL3_image-shared>,SDL3_image::SDL3_image-shared,SDL3_image::SDL3_image-static>
)