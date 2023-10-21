set(TARGET debug_conv_transpose)

add_executable(${TARGET} main.cpp)

install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE ggml encodec ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_14)

if(MSVC)
    target_compile_definitions(${TARGET} PRIVATE -D_CRT_SECURE_NO_WARNINGS=1)
endif()
