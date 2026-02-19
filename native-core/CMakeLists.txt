cmake_minimum_required(VERSION 3.20)
project(native_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(native_core STATIC
  src/native_core.cpp
)

target_include_directories(native_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(native_core PRIVATE
  -Wall -Wextra -Wpedantic
)

add_executable(native_core_bench
  src/main_bench.cpp
)

target_link_libraries(native_core_bench PRIVATE native_core)
