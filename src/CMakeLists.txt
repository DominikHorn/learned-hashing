cmake_minimum_required(VERSION 3.19)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -march=native -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -fsanitize=address,leak,undefined")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# ==== Non compatible compile flags ====
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-static-libsan HAS_LIBSAN)
if (HAS_LIBSAN)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -static-libsan")
else()
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -static-libasan")
endif()

# ==== Test target ====
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/96f4ce02a3a78d63981c67acbd368945d11d7d70.zip
  )

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
add_executable(lh_tests tests.cpp)
target_link_libraries(lh_tests learned-hashing gtest_main)

# enable ctest support (i.e., test discovery)
include(GoogleTest)
gtest_discover_tests(lh_tests)

# ==== Benchmark executable ====
add_executable(lh_benchmarks benchmarks.cpp)
target_link_libraries(lh_benchmarks PRIVATE learned-hashing ${GOOGLEBENCHMARK_LIBRARY})
