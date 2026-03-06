cmake_minimum_required(VERSION 3.12)
project(ImageROIAnalyzer)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# set(ncnn_DIR "/home/world/workspace/xitu3D/hi3516/zhaoyanxin/ncnn/lib/cmake/ncnn")
set(ncnn_DIR "/home/datas/workspace/xiezheng/wanxiaozhi/ncnn/ncnn-20250916/build/install/lib/cmake/ncnn")
# set(ncnn_DIR "/home/world/workspace/ncnn/ncnn-20241226/build/install/lib/cmake/ncnn")
# set(ncnn_DIR "/home/world/workspace/ncnn/ncnn-20250503/build/install/lib/cmake/ncnn")

find_package(ncnn REQUIRED)

# 添加可执行文件
add_executable(face_det_ncnn main.cpp)

# 链接数学库 (需要用于sqrt函数)
target_link_libraries(face_det_ncnn m ${OpenCV_LIBS} ncnn)

# 启用更严格的编译警告
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(face_det_ncnn PRIVATE -Wall -Wextra -pedantic)
endif()

# 如果使用Visual Studio，设置警告级别
if(MSVC)
    target_compile_options(face_det_ncnn PRIVATE /W4)
endif()

# 可选：添加测试目标
option(ENABLE_TESTS "Enable test building" OFF)
if(ENABLE_TESTS)
    enable_testing()
    add_test(NAME PalmNcnn_Test COMMAND face_det_ncnn)
endif()