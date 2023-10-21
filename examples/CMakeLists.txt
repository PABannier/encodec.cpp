add_library(common STATIC common.cpp)
target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(main)
add_subdirectory(encoder_only)
add_subdirectory(decoder_only)