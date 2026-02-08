cmake_minimum_required(VERSION 3.20)
project(SegMesh LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- GLFW (submodule) ----
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(external/glfw)

# ---- bgfx (submodule) ----
# Recent bgfx versions ship a CMakeLists.txt at repo root.
# If your bgfx revision does NOT, tell me and I'll switch you to the GENie build path.
set(BGFX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BGFX_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(BGFX_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(external/bgfx)

# ---- App ----
add_executable(segmesh
  src/main.cpp
)

target_include_directories(segmesh PRIVATE
  external/bgfx/include
  external/bx/include
  external/bimg/include
)

# Native window handle access for Wayland/X11
target_compile_definitions(segmesh PRIVATE
  GLFW_INCLUDE_NONE
  GLFW_EXPOSE_NATIVE_WAYLAND
  GLFW_EXPOSE_NATIVE_X11
)

target_link_libraries(segmesh PRIVATE
  glfw
  bgfx
  bx
  bimg
)

if(UNIX AND NOT APPLE)
  target_link_libraries(segmesh PRIVATE dl pthread)
endif()
