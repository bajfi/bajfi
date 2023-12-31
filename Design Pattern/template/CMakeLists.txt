cmake_minimum_required(VERSION 3.25)
project(template_method)

set(CMAKE_CXX_STANDARD 17)

add_executable(template_method main.cpp templateClass.h workflow.h)
