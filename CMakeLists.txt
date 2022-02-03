cmake_minimum_required(VERSION 3.0.0)
list(APPEND CMAKE_MESSAGE_CONTEXT ulqr)
project(ulqr VERSION 0.1.0 LANGUAGES C)

# Include useful modules
include(CMakePrintHelpers)
include(GNUInstallDirs)

##############################
# Options
##############################

# Handle default build type
set(ULQR_DEFAULT_BUILD_TYPE "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type specified. Setting CMAKE_BUILD_TYPE to ${ULQR_DEFAULT_BUILD_TYPE}")
  set(CMAKE_BUILD_TYPE ${ULQR_DEFAULT_BUILD_TYPE} CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Enable testing
option(ULQR_BUILD_TESTS "Build tests for ulqr" ON)

# Code Coverage
option(ULQR_CODE_COVERAGE "Compile rsLQR with Code Coverage." OFF)
if(ULQR_CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  message(STATUS "Compiling ulqr with coverage info.")
  add_compile_options(-O0 -g -fprofile-arcs -ftest-coverage)
  add_link_options(-fprofile-arcs -ftest-coverage)
endif()

# Documentation
option(ULQR_BUILD_DOCS "Build documentation for ulqr." OFF)

##############################
# Dependencies
##############################
if (ULQR_BUILD_DOCS)
  find_package(Doxygen REQUIRED)
endif()

##############################
# Build
##############################

# Compile options
add_compile_options(-Wall -Wextra -pedantic -Werror)

# Make all includes relative to src/ directory
include_directories(${PROJECT_SOURCE_DIR}/src)

# Build source files
add_subdirectory(src)

##############################
# Testing 
##############################
if (ULQR_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

##############################
# Documentation 
##############################
if (ULQR_BUILD_DOCS)
  add_subdirectory(docs)
endif()