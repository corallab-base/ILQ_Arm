cmake_minimum_required(VERSION 3.22)
project(ilqgames C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_BUILD_RPATH "/usr/local/lib")
set(CMAKE_INSTALL_RPATH "/usr/local/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Build options.
option(BUILD_TESTS "Build tests" ON)

# Add cmake modules.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
message("Cmake module path: ${CMAKE_MODULE_PATH}")

# Add cmake utilities.
include(cmake/Utils.cmake)

# Add dependencies.
include(cmake/Dependencies.cmake)

# Find Pinocchio
find_package(pinocchio REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${PINOCCHIO_INCLUDE_DIRS})

if (TARGET pinocchio::pinocchio_collision)
  message(STATUS "Pinocchio FCL/COAL: ON")
else()
  message(STATUS "Pinocchio FCL/COAL: OFF")
endif()

# Check for C++17 features and enable.
ilqgames_enable_cpp17()

# Set compiler constants.
add_definitions(-DSOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_definitions(-DILQGAMES_LOG_DIR="${CMAKE_SOURCE_DIR}/logs")
add_definitions(-DILQGAMES_TEST_DIR="${CMAKE_SOURCE_DIR}/random_tests")

add_definitions(-DIMGUI_IMPL_OPENGL_LOADER_GLEW)

# Set the build type. Default to Release mode.
if(NOT CMAKE_BUILD_TYPE)
  message("Defaulting to building targets in Release mode.")
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

# Find and build third party libraries.
add_subdirectory(external)

# Add the source directories to the search path so cmake can find our headers.
include_directories(${CMAKE_SOURCE_DIR}/include ${imgui_SOURCE_DIR}/include)

# Collect source files and build the ilqgames object.
file(GLOB_RECURSE ilqgames_srcs ${CMAKE_SOURCE_DIR}/src/*.cpp)
add_library(ilqgames ${ilqgames_srcs})

# Always link core; add collision only if that IMPORTED target exists
target_link_libraries(ilqgames PUBLIC pinocchio::pinocchio Eigen3::Eigen)
if (TARGET pinocchio::pinocchio_collision)
  target_link_libraries(ilqgames PUBLIC pinocchio::pinocchio_collision)
endif()

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ilqgames DESTINATION include/ilqgames)
install(TARGETS ilqgames
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

# Set up output directory for executables.
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin/)

# Find and build executables.
add_subdirectory(exec)

# Find and build tests.
if (BUILD_TESTS)
  message("Build tests is enabled.")
  add_subdirectory(test)
endif (BUILD_TESTS)