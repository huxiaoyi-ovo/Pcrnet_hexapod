cmake_minimum_required(VERSION 3.0.2)
project(servo)

file(GLOB hdrs *.h)
file(GLOB srs *.cpp)

add_library(SCServo STATIC ${hdrs} ${srs})