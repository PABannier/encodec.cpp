set(TARGET decoder_only)
add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE encodec common)
