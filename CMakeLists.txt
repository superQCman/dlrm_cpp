cmake_minimum_required(VERSION 3.10)
project(DLRM LANGUAGES CXX)

set(CMAKE_PREFIX_PATH "~/libtorch")  # LibTorch路径
find_package(Torch REQUIRED)
set(CMAKE_CXX_STANDARD 17) 
add_executable(DLRM dlrm.cpp)

target_link_libraries(DLRM PRIVATE ${TORCH_LIBRARIES} pthread)

set(CMAKE_BUILD_TYPE Debug)
