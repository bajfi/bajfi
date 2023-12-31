cmake_minimum_required(VERSION 3.25)
project(Observer)

set(CMAKE_CXX_STANDARD 17)

add_executable(Observer main.cpp subscriberBase.h subscribeClasses.h)
