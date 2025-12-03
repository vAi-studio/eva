cmake_minimum_required(VERSION 3.16)

project(eva VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build options
option(EVA_BUILD_SAMPLES "Build EVA sample projects" ON)

# Find Vulkan
find_package(Vulkan REQUIRED)
if (NOT Vulkan_FOUND)
    message(FATAL_ERROR "Vulkan not found. Please install the Vulkan SDK.")
endif()

# External dependencies directory
if(NOT DEFINED EVA_EXTERNAL_DIR)
    # Use eva's own external directory
    set(EVA_EXTERNAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external")
endif()



################################################################################
# GLFW (check if already configured by parent)
################################################################################
if(NOT TARGET glfw)
    message(STATUS "[EVA] Configuring GLFW...")
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "Build GLFW documentation")
    set(GLFW_INSTALL OFF CACHE BOOL "Generate installation target")
    add_subdirectory(${EVA_EXTERNAL_DIR}/glfw EXCLUDE_FROM_ALL)
    message(STATUS "[EVA] GLFW configuration completed.")
else()
    message(STATUS "[EVA] Using existing GLFW target")
endif()


################################################################################
# SPIRV-Reflect
################################################################################
if (Vulkan_FOUND)
    message(STATUS "[SPIRV-Reflect] Configuring SPIRV-Reflect...")
    set(SPIRV_REFLECT_EXECUTABLE OFF CACHE BOOL "Build spirv-reflect executable")
    set(SPIRV_REFLECT_STATIC_LIB ON CACHE BOOL "Build a SPIRV-Reflect static library")
    # set(SPIRV_REFLECT_EXAMPLES OFF CACHE BOOL "Build the SPIRV-Reflect example programs")
    # set(SPIRV_REFLECT_ENABLE_ASSERTS ON CACHE BOOL "Enable asserts for debugging")
    add_subdirectory(external/SPIRV-Reflect)
    message(STATUS "[SPIRV-Reflect] SPIRV-Reflect configuration completed.")
endif()


################################################################################
# EVA Library
################################################################################
# Create static library
add_library(eva STATIC
    src/eva-runtime.cpp
    src/eva-spirv-reflect.cpp

    src/eva-runtime.h
    src/eva-enums.h
    src/eva-error.h
    src/eva-native-factory.h
)

# Include directories
target_include_directories(eva
PRIVATE
    ${Vulkan_INCLUDE_DIRS}
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link libraries
target_link_libraries(eva
PRIVATE
    ${Vulkan_LIBRARIES}
    glfw
    spirv-reflect-static
)

# Compiler definitions
target_compile_definitions(eva
PRIVATE
    GLFW_INCLUDE_VULKAN
)

# Android
if(ANDROID)
    target_link_libraries(eva 
    PRIVATE
        android
    )
endif()


################################################################################
# Shader compilation utility function
################################################################################
# Find glslangValidator
set(EVA_GLSLC "${Vulkan_GLSLC_EXECUTABLE}")
if(EVA_GLSLC)
    message(STATUS "[EVA] Found glslc: ${EVA_GLSLC}")
else()
    message(WARNING "[EVA] glslc not found. Shader compilation will not be available.")
endif()

# Define shader file extensions
set(SHADER_EXTENSIONS
    .vert .frag .comp .geom .tesc .tese
    .rgen .rint .rahit .rchit .rmiss .rcall
    .mesh .task .glsl
)

# Register shader extensions as source file properties
if(EVA_GLSLC)
    foreach(EXT ${SHADER_EXTENSIONS})
        set_source_files_properties(
            ${CMAKE_CURRENT_SOURCE_DIR}/dummy${EXT}
            PROPERTIES
            LANGUAGE C
        )
    endforeach()

    # Create a function to automatically compile shaders added to targets
    # Usage: eva_auto_compile_shaders(target_name [output_dir])
    function(eva_auto_compile_shaders TARGET_NAME)
        # Optional second argument: output directory
        set(OUTPUT_DIR "${ARGV1}")

        get_target_property(TARGET_SOURCES ${TARGET_NAME} SOURCES)

        set(SHADER_FILES)
        foreach(SRC ${TARGET_SOURCES})
            get_filename_component(EXT ${SRC} LAST_EXT)
            if(EXT IN_LIST SHADER_EXTENSIONS)
                list(APPEND SHADER_FILES ${SRC})
            endif()
        endforeach()

        if(SHADER_FILES)
            # Remove shader files from target sources
            list(REMOVE_ITEM TARGET_SOURCES ${SHADER_FILES})
            set_property(TARGET ${TARGET_NAME} PROPERTY SOURCES ${TARGET_SOURCES})

            # Compile them using eva_compile_shaders
            if(OUTPUT_DIR)
                eva_compile_shaders(
                    TARGET ${TARGET_NAME}
                    OUTPUT_DIR ${OUTPUT_DIR}
                    SHADERS ${SHADER_FILES}
                )
            else()
                eva_compile_shaders(
                    TARGET ${TARGET_NAME}
                    SHADERS ${SHADER_FILES}
                )
            endif()

            list(LENGTH SHADER_FILES SHADER_COUNT)
            message(STATUS "[EVA] Auto-compiled ${SHADER_COUNT} shader(s) for ${TARGET_NAME}")
        endif()
    endfunction()
endif()

# Function to compile shaders
# Usage: eva_compile_shaders(TARGET target_name SHADERS shader1.rgen shader2.frag ...)
function(eva_compile_shaders)
    if(NOT EVA_GLSLC)
        message(WARNING "[EVA] glslc not found. Skipping shader compilation.")
        return()
    endif()

    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "TARGET;OUTPUT_DIR" "SHADERS")

    if(NOT ARG_TARGET)
        message(FATAL_ERROR "[EVA] eva_compile_shaders: TARGET argument is required")
    endif()

    if(NOT ARG_SHADERS)
        message(WARNING "[EVA] eva_compile_shaders: No shaders specified")
        return()
    endif()

    set(SHADER_OUTPUTS)
    foreach(SHADER_SOURCE ${ARG_SHADERS})
        # Get absolute path if relative
        get_filename_component(SHADER_SOURCE_ABS ${SHADER_SOURCE} ABSOLUTE)
        get_filename_component(SHADER_NAME ${SHADER_SOURCE_ABS} NAME)
        get_filename_component(SHADER_DIR ${SHADER_SOURCE_ABS} DIRECTORY)

        # Determine output path
        if(ARG_OUTPUT_DIR)
            # Use custom output directory
            set(SHADER_OUTPUT "${ARG_OUTPUT_DIR}/${SHADER_NAME}.spv")
        else()
            # Use same directory as source shader
            set(SHADER_OUTPUT "${SHADER_DIR}/${SHADER_NAME}.spv")
        endif()

        add_custom_command(
            OUTPUT ${SHADER_OUTPUT}
            COMMAND ${EVA_GLSLC} --target-env=vulkan1.2 -o ${SHADER_OUTPUT} ${SHADER_SOURCE_ABS}
            DEPENDS ${SHADER_SOURCE_ABS}
            COMMENT "[EVA] Compiling shader: ${SHADER_NAME} -> ${SHADER_NAME}.spv"
            VERBATIM
        )

        list(APPEND SHADER_OUTPUTS ${SHADER_OUTPUT})
    endforeach()

    # Create output directory if specified
    if(ARG_OUTPUT_DIR)
        file(MAKE_DIRECTORY ${ARG_OUTPUT_DIR})
    endif()

    # Create custom target
    set(SHADER_TARGET "${ARG_TARGET}_shaders")
    add_custom_target(${SHADER_TARGET} DEPENDS ${SHADER_OUTPUTS})

    # Add dependency to main target
    add_dependencies(${ARG_TARGET} ${SHADER_TARGET})

    message(STATUS "[EVA] Configured shader compilation for target: ${ARG_TARGET}")
endfunction()


################################################################################
# Samples
################################################################################
if(EVA_BUILD_SAMPLES)
    message(STATUS "[EVA] Building sample projects...")
    add_subdirectory(samples/raytracing)
    message(STATUS "[EVA] Sample projects build configuration completed.")
endif()