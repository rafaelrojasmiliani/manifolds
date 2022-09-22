cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(manifolds VERSION "0.0.1")
include(CTest)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -funroll-loops -mfpmath=sse")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -pthread ")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_INSTALL_PREFIX /usr)
set(PYTHON_VERSION
    3.6
    CACHE STRING "Build bindings for Python version")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# include ifopt with custon ROS installation
find_package(Eigen3 REQUIRED)
add_library(
  manifolds SHARED
  ${PROJECT_SOURCE_DIR}/src/Sphere.cpp ${PROJECT_SOURCE_DIR}/src/MapBase.cpp
  ${PROJECT_SOURCE_DIR}/src/MapBaseComposition.cpp #
)

target_include_directories(
  manifolds
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include> ${Eigen3_INCLUDE_DIRS}
         /usr/include/eigen3)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
  enable_testing()
  add_subdirectory(tests)
endif()

if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json")
  execute_process(
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different
      ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
      ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json)
endif()
