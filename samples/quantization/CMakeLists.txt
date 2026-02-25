cmake_minimum_required(VERSION 3.16)

project(sample-quantization)

file(GLOB SHADERS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.comp)

add_executable(${PROJECT_NAME}
    main.cpp
    ${SHADERS}
)

# Auto-detect and compile shaders
eva_auto_compile_shaders(${PROJECT_NAME} ${CMAKE_CURRENT_BINARY_DIR}/shaders/quantization)

add_definitions(
    -DPROJECT_CURRENT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    -DSHADER_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders/quantization"
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/samples/matmul
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    eva
)
