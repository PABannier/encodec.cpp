set(TARGET decompress)
add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE encodec common)
target_compile_features(${TARGET} PRIVATE cxx_std_11)