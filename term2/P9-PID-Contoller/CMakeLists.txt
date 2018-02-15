project(PID)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "${CXX_FLAGS}")

set(sources src/PID.cpp)
set(test_sources test/setup.cpp test/test_PID.cpp)

include_directories(/usr/local/include)
link_directories(/usr/local/lib)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

include_directories(/usr/local/opt/openssl/include)
link_directories(/usr/local/opt/openssl/lib)
link_directories(/usr/local/Cellar/libuv/1*/lib)

endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

# Add source files to a library to avoid additional compile for unit tests.
add_library (pid_lib ${sources})

# Target app
add_executable(pid src/main.cpp)
target_link_libraries(pid pid_lib z ssl uv uWS)

# Unit tests
add_executable(pid_unit_tests ${test_sources})
target_link_libraries(pid_unit_tests pid_lib)
enable_testing()
add_test(NAME PID_Test_Suite COMMAND pid_unit_tests)
