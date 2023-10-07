cmake_minimum_required(VERSION 3.25)

project(MatrixC CXX)

set(ENABLE_TESTS FALSE)

set(CMAKE_CXX_STANDARD 23)

include_directories(include)

set(HEADERS include/Matrix.h)

add_library(MatrixC INTERFACE ${HEADERS})

if (ENABLE_TESTS)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-Bstatic,--whole-archive -lwinpthread -Wl,--no-whole-archive")

    add_subdirectory(../GoogleTest gtest)

    include_directories(tests)

    set(TESTS tests/main.cpp tests/MatrixTests.cpp)

    add_executable(MatrixC_TEST ${TESTS})
    target_link_libraries(MatrixC_TEST -lwsock32 -lws2_32 MatrixC gtest pthread)
endif ()