cmake_minimum_required(VERSION 3.20)


set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_TOOLCHAIN_FILE}" CACHE STRING "Vcpkg toolchain file")

message(STATUS "CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")

# list(APPEND CMAKE_MODULE_PATH "$ENV{VCPKG_ROOT}/installed/x64-windows/share")

project(photon_loom)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Stb REQUIRED)
# # find_package(TinyObjLoader REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)

add_executable(photon_loom main.cpp)

if (Stb_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${Stb_INCLUDE_DIRS})
endif()

if (tinyobjloader_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${TINYOBJLOADER_INCLUDE_DIRS})
endif()
# 添加 include 路径
# Stb
# if (Stb_FOUND)
    
# endif()
# # TinyObjLoader
# if (TinyObjLoader_FOUND)
    
# endif()
# # 链接库（如有需要）
# if (Stb_FOUND)
#     target_link_libraries(${PROJECT_NAME} PRIVATE Stb::Stb)
# endif()
# if (TinyObjLoader_FOUND)
#     target_link_libraries(${PROJECT_NAME} PRIVATE TinyObjLoader::TinyObjLoader)
# endif()