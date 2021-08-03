cmake_minimum_required(VERSION 3.19)
project("learned-hashing" VERSION 1.0
        DESCRIPTION "A header only cmake/c++ library that exposes various state-of-the-art learned hash functions"
        HOMEPAGE_URL "https://github.com/DominikHorn/learned-hashing"
        LANGUAGES CXX)

# Declare library & directories to include. See
# http://mariobadr.com/creating-a-header-only-library-with-cmake.html for more
# info/install instructions
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)

# Make IDE friendly
target_sources(${PROJECT_NAME} INTERFACE learned_hashing.hpp include/)

# Benchmark code
include(${PROJECT_SOURCE_DIR}/thirdparty/googlebenchmark.cmake)
add_subdirectory(src)
