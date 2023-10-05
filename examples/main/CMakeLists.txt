set(TARGET main)

add_executable(${TARGET} main.cpp dr_wav.h)

install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE encodec.cpp ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_14)

if(MSVC)
    target_compile_definitions(${TARGET} PRIVATE -D_CRT_SECURE_NO_WARNINGS=1)
endif()
