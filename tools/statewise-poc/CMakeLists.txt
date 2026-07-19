set(TARGET llama-statewise-poc)
add_executable(${TARGET} statewise-poc.cpp)
target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
