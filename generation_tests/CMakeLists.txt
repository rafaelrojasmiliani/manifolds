cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(manifold_code_generation VERSION "0.0.1")
include(CTest)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -funroll-loops -mfpmath=sse")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -pthread")
# set(CMAKE_CXX_FLAGS_DEBUG "-g3 -pthread -fsanitize=address -static-libasan ")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED)
file(GLOB Sources "${PROJECT_SOURCE_DIR}/src/*.cpp")
add_library(manifold_code_generation SHARED ${Sources})

target_include_directories(
  manifold_code_generation
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include> ${Eigen3_INCLUDE_DIRS}
         /usr/include/eigen3)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
  enable_testing()
  add_subdirectory(tests)
endif()

if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
  ")
  execute_process(
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different
      ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
      ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json)
endif()
