include(GoogleTest)
include_directories(PUBLIC ${PROJECT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIRS}
                    ${pinocchio_INCLUDE_DIRS} ${ifopt_INCLUDE_DIRS})

file(GLOB_RECURSE test_list ${PROJECT_SOURCE_DIR}/tests/*.cpp)
file(GLOB_RECURSE test_py_list ${CMAKE_SOURCE_DIR}/tests/*.py)
list(REMOVE_ITEM test_py_list "${PROJECT_SOURCE_DIR}/tests/__init__.py")
list(REMOVE_ITEM test_py_list "${PROJECT_SOURCE_DIR}/tests/tools.py")
if(NOT USE_PINOCCHIO)
  list(REMOVE_ITEM test_list "${PROJECT_SOURCE_DIR}/tests/pinocchio.cpp")
endif()

# https://developers.redhat.com/blog/2021/05/05/memory-error-checking-in-c-and-c-comparing-sanitizers-and-valgrind
set(ENV_ASAN
    ASAN_OPTIONS=abort_on_error=1:fast_unwind_on_malloc=1:detect_leaks=1;
    UBSAN_OPTIONS=print_stacktrace=1; G_SLICE=always-malloc;
    G_DEBUG=gc-friendly)

foreach(file_path ${test_list})
  get_filename_component(barename ${file_path} NAME)
  string(REPLACE ".cpp" "" new_name ${barename})
  add_executable(${new_name} ${file_path})
  target_compile_options(${new_name} PRIVATE -Wall -Wextra)
  gtest_discover_tests(
    ${new_name}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests
    PROPERTIES ENVIRONMENT ${ENV_ASAN})

  # set_target_properties(your_target_name PROPERTIES WORKING_DIRECTORY
  # "/path/to/your/working/directory")

  # https://github.com/stack-of-tasks/pinocchio/issues/1261
  target_link_libraries(
    ${new_name} PRIVATE gtest pthread manifolds Eigen3::Eigen
                        pinocchio::pinocchio ${ifopt_LIBRARIES})
endforeach()

foreach(file_path ${test_py_list})
  get_filename_component(barename ${file_path} NAME)
  string(REPLACE ".py" "_py" new_name ${barename})
  string(REPLACE ".py" "" mod_name ${barename})
  add_test(
    NAME ${new_name}
    COMMAND python3 -m unittest tests.${mod_name}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  set_tests_properties(
    ${new_name}
    PROPERTIES
      ENVIRONMENT
      PYTHONPATH=${PROJECT_SOURCE_DIR}/python:${PROJECT_BINARY_DIR}:$ENV{PYTHONPATH}
  )

endforeach()
