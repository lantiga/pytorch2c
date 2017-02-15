project(foo)

set(SRCS mnist_forward_test.c)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/install/include/TH)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/install/include/THNN)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/install/lib)

add_executable(mnist_forward_test ${SRCS})
target_link_libraries(mnist_forward_test TH THNN)

