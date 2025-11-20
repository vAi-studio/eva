cmake_minimum_required(VERSION 3.16)

project(vulkan-engine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Library source files
set(VE_SOURCES
    src/eva-runtime.cpp
    src/spirvHelpers.cpp
)

set(VE_HEADERS
    src/eva-runtime.h
    src/error.h
    src/eva-native-factory.h
)

# Create static library
add_library(ve STATIC ${VE_SOURCES} ${VE_HEADERS})

# Find Vulkan
find_package(Vulkan REQUIRED)
if (NOT Vulkan_FOUND)
    message(FATAL_ERROR "Vulkan not found. Please install the Vulkan SDK.")
endif()

# External dependencies directory
if(NOT DEFINED VE_EXTERNAL_DIR)
    # Use ve's own external directory
    set(VE_EXTERNAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external")
endif()

################################################################################
# SPIRV-Tools (check if already configured by parent)
################################################################################
if(NOT TARGET SPIRV-Tools)
    message(STATUS "[VE] Configuring SPIRV-Tools...")
    set(SPIRV-Headers_SOURCE_DIR "${VE_EXTERNAL_DIR}/spirv-headers")
    set(mimalloc_SOURCE_DIR "${VE_EXTERNAL_DIR}/mimalloc")
    add_subdirectory(${VE_EXTERNAL_DIR}/SPIRV-Tools EXCLUDE_FROM_ALL)
    message(STATUS "[VE] SPIRV-Tools configuration completed.")
else()
    message(STATUS "[VE] Using existing SPIRV-Tools target")
endif()

################################################################################
# SPIRV-Reflect (check if already configured by parent)
################################################################################
if(NOT TARGET spirv-reflect-static)
    message(STATUS "[VE] Configuring SPIRV-Reflect...")
    set(SPIRV_REFLECT_EXECUTABLE OFF CACHE BOOL "Build spirv-reflect executable")
    set(SPIRV_REFLECT_STATIC_LIB ON CACHE BOOL "Build a SPIRV-Reflect static library")
    add_subdirectory(${VE_EXTERNAL_DIR}/SPIRV-Reflect EXCLUDE_FROM_ALL)
    message(STATUS "[VE] SPIRV-Reflect configuration completed.")
else()
    message(STATUS "[VE] Using existing SPIRV-Reflect target")
endif()

################################################################################
# glslang (check if already configured by parent)
################################################################################
if(NOT TARGET glslang)
    message(STATUS "[VE] Configuring glslang...")
    set(BUILD_EXTERNAL OFF CACHE BOOL "Disable building external dependencies")
    set(ALLOW_EXTERNAL_SPIRV_TOOLS ON CACHE BOOL "Allow external SPIRV-Tools")
    set(ENABLE_HLSL OFF CACHE BOOL "Disable HLSL support")
    set(ENABLE_RTTI ON CACHE BOOL "Enable RTTI for glslang")
    set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "Disable glslang binaries")
    set(SKIP_SPIRV_TOOLS_INSTALL ON CACHE BOOL "Skip installation of SPIRV-Tools")
    set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "Skip building SPIRV-Tools executables")
    add_subdirectory(${VE_EXTERNAL_DIR}/glslang EXCLUDE_FROM_ALL)
    message(STATUS "[VE] glslang configuration completed.")
else()
    message(STATUS "[VE] Using existing glslang target")
endif()

################################################################################
# GLFW (check if already configured by parent)
################################################################################
if(NOT TARGET glfw)
    message(STATUS "[VE] Configuring GLFW...")
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "Build GLFW documentation")
    set(GLFW_INSTALL OFF CACHE BOOL "Generate installation target")
    add_subdirectory(${VE_EXTERNAL_DIR}/glfw EXCLUDE_FROM_ALL)
    message(STATUS "[VE] GLFW configuration completed.")
else()
    message(STATUS "[VE] Using existing GLFW target")
endif()

# Include directories
target_include_directories(ve
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        ${Vulkan_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(ve
    PUBLIC
        ${Vulkan_LIBRARIES}
        glfw
    PRIVATE
        spirv-reflect-static
        glslang
        glslang-default-resource-limits
)

# Compiler definitions
target_compile_definitions(ve
    PUBLIC
        GLFW_INCLUDE_VULKAN
)


################################################################################
# Platform-specific settings
################################################################################
# Android
if(ANDROID)
    target_link_libraries(ve PRIVATE
        android
    )
endif()