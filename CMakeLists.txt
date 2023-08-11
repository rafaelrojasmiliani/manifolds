cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(manifolds VERSION "0.0.1")
include(CTest)

set(EXTRA_CFLAGS=-fdiagnostics-color=always)

# GCCFilter, if appliciable if(CMAKE_COMPILER_IS_GNUCXX OR
# CMAKE_COMPILER_IS_GNUCPP) option(COLOR_GCC "Use GCCFilter to color compiler
# output messages" ON) set(COLOR_GCC_OPTIONS "-c -r -w" CACHE STRING "Arguments
# that are passed to gccfilter when output coloring is switchend on. Defaults to
# -c -r -w." ) if(COLOR_GCC) set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE
# "gccfilter ${COLOR_GCC_OPTIONS}") endif() endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -funroll-loops -mfpmath=sse")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -pthread ")

find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(ENV{ASAN_OPTIONS} "symbolize=1:detect_leaks=1")
  set(CMAKE_CXX_FLAGS_DEBUG
      "${CMAKE_CXX_FLAGS_DEBUG} \
      -fsanitize=address \
      -fsanitize=float-divide-by-zero \
      -fsanitize=float-cast-overflow \
      -fsanitize=undefined \
      -fno-sanitize=alignment \
      -fno-sanitize=null \
      -fno-omit-frame-pointer \
      -fno-sanitize-recover=all ")
  # -Q \ -ftime-report command for compilation profiling
  set(CMAKE_LINKER_FLAGS_DEBUG ${CMAKE_LINKER_FLAGS_DEBUG} -fsanitize=address)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_INSTALL_PREFIX /usr)
option(USE_PINOCCHIO "Enable Pinocchio" on)

set(PYTHON_VERSION
    3.6
    CACHE STRING "Build bindings for Python version")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# include ifopt with custon ROS installation
find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)

set(SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/Atlases/Sphere.cpp
    ${PROJECT_SOURCE_DIR}/src/MapBase.cpp
    ${PROJECT_SOURCE_DIR}/src/MapBaseComposition.cpp #
)

if(USE_PINOCCHIO)
  set(pinocchio_DIR /opt/openrobots/lib/cmake/pinocchio)
  set(hpp-fcl_DIR /opt/openrobots/lib/cmake/hpp-fcl)
  set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
  find_package(pinocchio REQUIRED)
endif()

add_library(manifolds SHARED ${SOURCE_FILES})
target_link_libraries(manifolds ${pinocchio_LIBRARIES})

target_include_directories(
  manifolds
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>
         ${Eigen3_INCLUDE_DIRS}
         /usr/include/eigen3
         ${PROJECT_SOURCE_DIR}/submodules/gcem/include/
         ${PROJECT_SOURCE_DIR}/submodules/fpm/include/
         ${pinocchio_INCLUDE_DIRS})

target_compile_options(manifolds PRIVATE -Wall -Wextra -Wpedantic -Werror)

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

pybind11_add_module(pymanifolds
                    ${PROJECT_SOURCE_DIR}/python/bindings/bindings.cpp)
target_include_directories(
  pymanifolds
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>
         ${Eigen3_INCLUDE_DIRS}
         /usr/include/eigen3
         ${PROJECT_SOURCE_DIR}/submodules/gcem/include/
         ${PROJECT_SOURCE_DIR}/submodules/fpm/include/
         ${pinocchio_INCLUDE_DIRS})
target_link_libraries(pymanifolds PUBLIC manifolds)
