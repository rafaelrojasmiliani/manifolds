include(GoogleTest)
link_directories(${PROJECT_SOURCE_DIR}/build)
link_libraries(gtest pthread)
include_directories(PUBLIC ${PROJECT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIRS})

file(GLOB_RECURSE test_list ${PROJECT_SOURCE_DIR}/tests/*.cpp)
file(GLOB_RECURSE test_py_list ${CMAKE_SOURCE_DIR}/tests/*.py)
list(REMOVE_ITEM test_py_list "${PROJECT_SOURCE_DIR}/tests/__init__.py")
list(REMOVE_ITEM test_py_list "${PROJECT_SOURCE_DIR}/tests/tools.py")

foreach(file_path ${test_list})
  get_filename_component(barename ${file_path} NAME)
  string(REPLACE ".cpp" "" new_name ${barename})
  add_executable(${new_name} ${file_path})
  gtest_discover_tests(${new_name})
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
      PYTHONPATH=${PROJECT_SOURCE_DIR}/python:${PROJECT_SOURCE_DIR}/build:$ENV{PYTHONPATH}
  )

endforeach()