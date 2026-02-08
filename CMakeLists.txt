cmake_minimum_required(VERSION 3.20)
project(segmesh LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------------------
# Optional: update submodules during configure
# -------------------------
find_package(Git QUIET)
option(GIT_SUBMODULE "Update git submodules during configure" ON)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git" AND GIT_SUBMODULE)
  message(STATUS "Updating submodules (--init --recursive)")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE GIT_SUBMOD_RESULT
  )
  if(NOT GIT_SUBMOD_RESULT EQUAL 0)
    message(FATAL_ERROR "git submodule update failed with ${GIT_SUBMOD_RESULT}")
  endif()
endif()

foreach(dep IN ITEMS bgfx bx bimg glfw)
  if(NOT EXISTS "${PROJECT_SOURCE_DIR}/submods/${dep}")
    message(FATAL_ERROR "Missing submodule: submods/${dep}. Did you run: git submodule update --init --recursive ?")
  endif()
endforeach()

# -------------------------
# GLFW (build from submodule, Wayland support is decided by system deps)
# -------------------------
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(submods/glfw)

# -------------------------
# bgfx / bx / bimg headers
# -------------------------
set(BGFX_DIR "${PROJECT_SOURCE_DIR}/submods/bgfx")
set(BX_DIR   "${PROJECT_SOURCE_DIR}/submods/bx")
set(BIMG_DIR "${PROJECT_SOURCE_DIR}/submods/bimg")

# -------------------------
# Find prebuilt bgfx libs (built outside CMake)
# -------------------------
set(BGFX_BIN_DIR "" CACHE PATH "Path to bgfx .build/*/bin directory containing libbgfx*.a")

if(BGFX_BIN_DIR STREQUAL "")
  # Search typical bgfx build outputs:
  file(GLOB_RECURSE _BGFX_CANDIDATES
    "${BGFX_DIR}/.build/*/bin/libbgfx*.a"
  )
  if(_BGFX_CANDIDATES)
    # Pick the first match and derive bin dir
    list(GET _BGFX_CANDIDATES 0 _FIRST_BGFX_LIB)
    get_filename_component(BGFX_BIN_DIR "${_FIRST_BGFX_LIB}" DIRECTORY)
    message(STATUS "Auto-detected BGFX_BIN_DIR: ${BGFX_BIN_DIR}")
  endif()
endif()

if(BGFX_BIN_DIR STREQUAL "" OR NOT EXISTS "${BGFX_BIN_DIR}")
  message(FATAL_ERROR
    "Could not detect bgfx build output.\n"
    "Build bgfx first so libs exist under submods/bgfx/.build/.../bin, or set:\n"
    "  -DBGFX_BIN_DIR=/full/path/to/submods/bgfx/.build/<something>/bin"
  )
endif()

# Map CMake build type -> bgfx suffix (Debug/Release)
set(_BGFX_SUFFIX "Release")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(_BGFX_SUFFIX "Debug")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  set(_BGFX_SUFFIX "Release")
elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
  set(_BGFX_SUFFIX "Release")
endif()

# Find the exact libraries. Names vary by bgfx revision/config;
find_library(BGFX_LIB NAMES bgfx${_BGFX_SUFFIX} bgfx PATHS "${BGFX_BIN_DIR}" NO_DEFAULT_PATH)
find_library(BX_LIB   NAMES bx${_BGFX_SUFFIX}   bx   PATHS "${BGFX_BIN_DIR}" NO_DEFAULT_PATH)
find_library(BIMG_LIB NAMES bimg${_BGFX_SUFFIX} bimg PATHS "${BGFX_BIN_DIR}" NO_DEFAULT_PATH)

if(NOT BGFX_LIB OR NOT BX_LIB OR NOT BIMG_LIB)
  message(FATAL_ERROR
    "Could not find one or more bgfx libs in: ${BGFX_BIN_DIR}\n"
    "Expected something like libbgfx${_BGFX_SUFFIX}.a, libbx${_BGFX_SUFFIX}.a, libbimg${_BGFX_SUFFIX}.a\n"
    "Try building bgfx (Debug/Release) and/or set BGFX_BIN_DIR explicitly."
  )
endif()

# Create imported targets
add_library(bgfx STATIC IMPORTED GLOBAL)
set_target_properties(bgfx PROPERTIES IMPORTED_LOCATION "${BGFX_LIB}")
target_include_directories(bgfx INTERFACE "${BGFX_DIR}/include")

add_library(bx STATIC IMPORTED GLOBAL)
set_target_properties(bx PROPERTIES IMPORTED_LOCATION "${BX_LIB}")
target_include_directories(bx INTERFACE "${BX_DIR}/include")

add_library(bimg STATIC IMPORTED GLOBAL)
set_target_properties(bimg PROPERTIES IMPORTED_LOCATION "${BIMG_LIB}")
target_include_directories(bimg INTERFACE "${BIMG_DIR}/include")

# -------------------------
# main app
# -------------------------
add_executable(segmesh
  src/main.cpp
)

target_include_directories(segmesh PRIVATE
  "${BGFX_DIR}/include"
  "${BX_DIR}/include"
  "${BIMG_DIR}/include"
)

# Native window handle access for Wayland (and optional X11 fallback)
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

find_package(PkgConfig REQUIRED)
pkg_check_modules(WAYLAND_CLIENT REQUIRED wayland-client)
target_include_directories(segmesh PRIVATE ${WAYLAND_CLIENT_INCLUDE_DIRS})
target_link_libraries(segmesh PRIVATE ${WAYLAND_CLIENT_LIBRARIES})


if(UNIX AND NOT APPLE)
  target_link_libraries(segmesh PRIVATE dl pthread)
endif()
