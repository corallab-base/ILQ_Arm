# If testing is enabled, build all tests.
if (BUILD_TESTS)
  enable_testing()

  # Set the local gtest include directory.
  set(GTEST_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/gtest")
  include_directories(${gtest_SOURCE_DIR} ${gtest_SOURCE_DIR}/include)

  # Gtest bug fix.
  add_definitions(-DGTEST_HAS_TR1_TUPLE=0)

  # Set a name for the output binary and testing arguments.
  set(test_target run_tests)
  set(test_args --gtest_shuffle)

  # 1. Glob all CPP files
  file(GLOB test_srcs ${CMAKE_SOURCE_DIR}/test/*.cpp)

  # 2. EXCLUDE the utility script from the test sources
  list(REMOVE_ITEM test_srcs "${CMAKE_SOURCE_DIR}/test/gen_random_start_goal.cpp")
  list(REMOVE_ITEM test_srcs "${CMAKE_SOURCE_DIR}/test/gen_complex_env_start_goal.cpp")

  foreach(test ${test_srcs})
    get_filename_component(test_no_ext ${test} NAME_WE)
    message("Including test   \"${BoldBlue}${test_no_ext}${ColorReset}\".")
  endforeach()

  # 3. Build the main test runner (WITHOUT the generator script)
  add_executable(${test_target} ${test_srcs})
  target_link_libraries(${test_target} ilqgames ${ilqgames_LIBRARIES} gtest)
  add_test(${test_target} ${test_target})
  ilqgames_set_runtime_directory(${test_target} ${PROJECT_BINARY_DIR})

  # 4. Build the generator script as its OWN standalone executable
  add_executable(gen_random_start_goal "${CMAKE_SOURCE_DIR}/test/gen_random_start_goal.cpp")
  target_link_libraries(gen_random_start_goal ilqgames ${ilqgames_LIBRARIES})
  add_executable(gen_complex_env_start_goal "${CMAKE_SOURCE_DIR}/test/gen_complex_env_start_goal.cpp")
  target_link_libraries(gen_complex_env_start_goal ilqgames ${ilqgames_LIBRARIES})

  # Make "make check" run the tests.
  add_custom_target(check COMMAND "${PROJECT_BINARY_DIR}/${test_target}")
  add_dependencies(check ${test_target})

  # Define a variable storing the path to test data.
  add_definitions(-DILQGAMES_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/test/test_data")

  # If you want a test to generate temporary files (ie, testing file io) then store
  # them in this ignored folder
  add_definitions(-DGENERATED_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/test/test_data/generated")
endif(BUILD_TESTS)
