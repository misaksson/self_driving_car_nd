project(PARTICLE_FILTER)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

set(sources src/particle_filter.cpp)
set(test_sources test/setup.cpp test/test_helper_functions.cpp test/test_particle_filter.cpp)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

include_directories(/usr/local/include)
include_directories(/usr/local/opt/openssl/include)
link_directories(/usr/local/lib)
link_directories(/usr/local/opt/openssl/lib)
link_directories(/usr/local/Cellar/libuv/1*/lib)

endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

# Add source files to a library to avoid additional compile for unit tests.
add_library (particle_filter_lib ${sources})

# Target app
add_executable(particle_filter src/main.cpp)
target_link_libraries(particle_filter particle_filter_lib z ssl uv uWS)

# Unit tests
add_executable(particle_filter_unit_tests ${test_sources})
target_link_libraries(particle_filter_unit_tests particle_filter_lib)
enable_testing()
add_test(NAME Particle_Filter_Test_Suite COMMAND particle_filter_unit_tests)
