cmake_minimum_required(VERSION 3.28)
project(ABCBackprop)

set(CMAKE_CXX_STANDARD 17)

add_executable(ABCBackprop ABCBackprop.cpp
        Parser.cpp)
