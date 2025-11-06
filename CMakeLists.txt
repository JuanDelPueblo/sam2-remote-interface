cmake_minimum_required(VERSION 3.16)
project(DataEngine VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add source and header directories
include_directories(include)

# Add subdirectories
add_subdirectory(src)