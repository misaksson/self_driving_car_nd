project(Path_Planning)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

set(sources src/constants.cpp src/helpers.cpp src/path/logic.cpp src/path/planner.cpp src/path/predict.cpp src/path/trajectory.cpp)
set(test_sources test/setup.cpp test/path/test_logic.cpp test/path/test_planner.cpp test/path/test_predict.cpp test/path/test_trajectory.cpp)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

include_directories(/usr/local/include)
include_directories(/usr/local/opt/openssl/include)
link_directories(/usr/local/lib)
link_directories(/usr/local/opt/openssl/lib)
link_directories(/usr/local/Cellar/libuv/1.11.0/lib)

endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

# Add source files to a library to avoid additional compile for unit tests.
add_library (path_planning_lib ${sources})

# Target app
add_executable(path_planning src/main.cpp)
target_link_libraries(path_planning path_planning_lib z ssl uv uWS)

# Unit tests
add_executable(path_planning_unit_tests ${test_sources})
target_link_libraries(path_planning_unit_tests path_planning_lib)
