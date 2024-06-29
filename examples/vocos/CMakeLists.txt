set(TARGET vocos)
add_executable(${TARGET} vocos.cpp)
target_link_libraries(${TARGET} PRIVATE encodec common ggml)
target_compile_features(${TARGET} PRIVATE cxx_std_11)
