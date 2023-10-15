cmake_minimum_required(VERSION 3.25)

project(MatrixC LANGUAGES CXX)

set(ENABLE_TESTS FALSE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

set(HEADERS include/Matrix.h)

add_library(MatrixC INTERFACE ${HEADERS})

if (ENABLE_TESTS)
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O0")
    else()
        set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O3")
    endif()

    add_subdirectory(../GoogleTest gtest)

    include_directories(tests)

    set(TESTS tests/main.cpp tests/MatrixTests.cpp)

    add_executable(MatrixC_TEST ${TESTS})
    target_link_libraries(MatrixC_TEST MatrixC gtest)
endif ()