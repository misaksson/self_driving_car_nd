project(ExtendedKF)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

set(sources src/tools.cpp src/FusionEKF.cpp src/kalman_filter.cpp src/tools.h src/FusionEKF.h src/kalman_filter.h)
set(test_sources test/setup.cpp test/test_tools.cpp test/test_kalman_filter.cpp test/test_FusionEKF.cpp)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

include_directories(/usr/local/include)
include_directories(/usr/local/opt/openssl/include)
link_directories(/usr/local/lib)
link_directories(/usr/local/opt/openssl/lib)
link_directories(/usr/local/Cellar/libuv/1*/lib)

endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

# Add source files to a library to avoid additional compile for unit tests.
add_library (ExtendedKF_lib ${sources})

# Target app
add_executable(ExtendedKF src/main.cpp)
target_link_libraries(ExtendedKF ExtendedKF_lib z ssl uv uWS)

# Unit tests
add_executable(ExtendedKF_unit_tests ${test_sources})
target_link_libraries(ExtendedKF_unit_tests ExtendedKF_lib)
enable_testing()
add_test(NAME ExtendedKF_Test_Suite COMMAND ExtendedKF_unit_tests)
