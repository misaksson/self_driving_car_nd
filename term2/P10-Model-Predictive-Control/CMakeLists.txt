project(MPC)

cmake_minimum_required (VERSION 3.5)

# -g allows for gdb debugging
# turn on -03 for best performance
add_definitions(-std=c++11 -O3)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "${CXX_FLAGS}")

set(sources src/MPC.cpp src/path.cpp src/polynomial.cpp)
set(test_sources test/setup.cpp test/test_path.cpp test/test_polynomial.cpp)

include_directories(/usr/local/include)
link_directories(/usr/local/lib)
include_directories(src/Eigen-3.3)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

include_directories(/usr/local/opt/openssl/include)
link_directories(/usr/local/opt/openssl/lib)
link_directories(/usr/local/Cellar/libuv/1*/lib)

endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

# Add source files to a library to avoid additional compile for unit tests.
add_library (mpc_lib ${sources})

# Target app
add_executable(mpc src/main.cpp)
target_link_libraries(mpc mpc_lib ipopt z ssl uv uWS)

# Unit tests
add_executable(mpc_unit_tests ${test_sources})
target_link_libraries(mpc_unit_tests mpc_lib)
