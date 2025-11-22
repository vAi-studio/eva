cmake_minimum_required(VERSION 3.16)

project(raytracing)

add_executable(${PROJECT_NAME}
    # client code
    main.cpp

    # general
    math_types.h
    input.h
    camera.h

    # shaders (auto-compiled)
    shaders/raygen.rgen
    shaders/miss.rmiss
    shaders/chit.rchit
)

# Auto-detect and compile shaders
eva_auto_compile_shaders(${PROJECT_NAME} ${CMAKE_CURRENT_BINARY_DIR}/shaders)

add_definitions(
    -DPROJECT_CURRENT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    -DSHADER_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders"
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    eva
)
