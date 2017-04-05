cmake_minimum_required(VERSION 3.0)

project(torch2ctest)

include_directories(${INSTALL_DIR}/include/TH)
include_directories(${INSTALL_DIR}/include/THNN)
include_directories(${INSTALL_DIR}/include/torch2c)

link_directories(${INSTALL_DIR}/lib)

add_executable(${EXECUTABLE} ${SRC})
target_link_libraries(${EXECUTABLE} TH THNN)

