cmake_minimum_required(VERSION 3.15...3.29)
project(example LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(example neuro/codegen/bindings.cpp)
install(TARGETS example DESTINATION .)