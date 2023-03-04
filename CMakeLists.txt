cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(RC4_crack_cuda LANGUAGES C CXX CUDA)

include_directories(wingetopt)
add_subdirectory(wingetopt)
add_subdirectory(rc4_cuda)