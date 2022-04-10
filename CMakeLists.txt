cmake_minimum_required(VERSION 3.19)
project("learned-hashing" VERSION 1.0
        DESCRIPTION "A header only cmake/c++ library that exposes various state-of-the-art learned hash functions"
        HOMEPAGE_URL "https://github.com/DominikHorn/learned-hashing"
        LANGUAGES CXX)

# Declare library & directories to include. See
# http://mariobadr.com/creating-a-header-only-library-with-cmake.html for more
# info/install instructions
add_library(${PROJECT_NAME} INTERFACE)
target_sources(${PROJECT_NAME} INTERFACE learned_hashing.hpp include/)
target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

# Enable OpenMP if available
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-fopenmp HAS_OPENMP)
if (HAS_OPENMP)
  target_compile_options(${PROJECT_NAME} INTERFACE -fopenmp)
endif()

# ==== Dependencies ====
include(FetchContent)

FetchContent_Declare(
  pgm
  GIT_REPOSITORY https://github.com/DominikHorn/PGM-index 
  GIT_TAG 27bb71a
  )

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(BUILD_EXAMPLES OFF)
set(BUILD_PGM_TUNER OFF)
set(BUILD_PGM_BENCHMARK OFF)
FetchContent_MakeAvailable(pgm)

target_link_libraries(${PROJECT_NAME} INTERFACE pgmindexlib)

# Benchmark code
get_directory_property(hasParent PARENT_DIRECTORY)
if (NOT hasParent)
  # ==== Dependencies ====
  include(thirdparty/googlebenchmark.cmake)

  add_subdirectory(src)
endif()
