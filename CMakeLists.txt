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

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenMesh REQUIRED)

# -------------------------
# bgfx / bx / bimg headers
# -------------------------
set(BGFX_DIR "${PROJECT_SOURCE_DIR}/submods/bgfx")
set(BX_DIR   "${PROJECT_SOURCE_DIR}/submods/bx")
set(BIMG_DIR "${PROJECT_SOURCE_DIR}/submods/bimg")

set(SHADER_DIR "${PROJECT_SOURCE_DIR}/assets/shaders")

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

find_program(SHADERC
  NAMES shaderc${_BGFX_SUFFIX} shadercRelease shaderc
  PATHS "${BGFX_BIN_DIR}"
  NO_DEFAULT_PATH
)
if(NOT SHADERC)
  message(FATAL_ERROR "Could not find shaderc in: ${BGFX_BIN_DIR}")
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
  src/mesh_types.cpp
  src/obj_loader.cpp
  src/renderer.cpp
  src/segmentation.cpp
  src/ui.cpp
  src/main.cpp
  submods/bgfx/3rdparty/dear-imgui/imgui.cpp
  submods/bgfx/3rdparty/dear-imgui/imgui_draw.cpp
  submods/bgfx/3rdparty/dear-imgui/imgui_tables.cpp
  submods/bgfx/3rdparty/dear-imgui/imgui_widgets.cpp
  submods/bgfx/examples/common/imgui/imgui.cpp
)

set(SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders/spirv")
set(VERTEX_SHADER_OUTPUT "${SHADER_OUTPUT_DIR}/vs_triangle.bin")
set(FRAGMENT_SHADER_OUTPUT "${SHADER_OUTPUT_DIR}/fs_triangle.bin")

add_custom_command(
  OUTPUT "${VERTEX_SHADER_OUTPUT}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_OUTPUT_DIR}"
  COMMAND "${SHADERC}"
    -f "${SHADER_DIR}/vs_triangle.sc"
    -o "${VERTEX_SHADER_OUTPUT}"
    --type vertex
    --platform linux
    --profile spirv
    --varyingdef "${SHADER_DIR}/varying.def.sc"
    -i "${BGFX_DIR}/src"
    -i "${BGFX_DIR}/examples/common"
  DEPENDS
    "${SHADER_DIR}/vs_triangle.sc"
    "${SHADER_DIR}/varying.def.sc"
  VERBATIM
)

add_custom_command(
  OUTPUT "${FRAGMENT_SHADER_OUTPUT}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_OUTPUT_DIR}"
  COMMAND "${SHADERC}"
    -f "${SHADER_DIR}/fs_triangle.sc"
    -o "${FRAGMENT_SHADER_OUTPUT}"
    --type fragment
    --platform linux
    --profile spirv
    --varyingdef "${SHADER_DIR}/varying.def.sc"
    -i "${BGFX_DIR}/src"
    -i "${BGFX_DIR}/examples/common"
  DEPENDS
    "${SHADER_DIR}/fs_triangle.sc"
    "${SHADER_DIR}/varying.def.sc"
  VERBATIM
)

add_custom_target(segmesh_shaders
  DEPENDS
    "${VERTEX_SHADER_OUTPUT}"
    "${FRAGMENT_SHADER_OUTPUT}"
)

add_dependencies(segmesh segmesh_shaders)

target_include_directories(segmesh PRIVATE
  "${BGFX_DIR}/include"
  "${BGFX_DIR}/3rdparty"
  "${BGFX_DIR}/examples/common"
  "${BGFX_DIR}/examples/common/imgui"
  "${BX_DIR}/include"
  "${BIMG_DIR}/include"
)
if(OPENMESH_INCLUDE_DIRS)
  target_include_directories(segmesh PRIVATE ${OPENMESH_INCLUDE_DIRS})
endif()

# Native window handle access for Linux backends.
target_compile_definitions(segmesh PRIVATE
  GLFW_INCLUDE_NONE
  $<$<CONFIG:Debug>:BX_CONFIG_DEBUG=1>
  $<$<NOT:$<CONFIG:Debug>>:BX_CONFIG_DEBUG=0>
)

target_link_libraries(segmesh PRIVATE
  glfw
  bgfx
  bx
  bimg
  Eigen3::Eigen
)
if(TARGET OpenMeshCore)
  target_link_libraries(segmesh PRIVATE OpenMeshCore)
  if(TARGET OpenMeshTools)
    target_link_libraries(segmesh PRIVATE OpenMeshTools)
  endif()
elseif(OPENMESH_LIBRARIES)
  target_link_libraries(segmesh PRIVATE ${OPENMESH_LIBRARIES})
else()
  message(FATAL_ERROR "OpenMesh was found, but CMake did not expose OpenMesh link targets/libraries.")
endif()

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(WAYLAND_CLIENT QUIET wayland-client)
endif()

find_package(X11 QUIET)

set(SEGMESH_HAS_LINUX_WINDOW_BACKEND OFF)

if(WAYLAND_CLIENT_FOUND)
  message(STATUS "Including Wayland support")
  target_compile_definitions(segmesh PRIVATE GLFW_EXPOSE_NATIVE_WAYLAND)
  target_include_directories(segmesh PRIVATE ${WAYLAND_CLIENT_INCLUDE_DIRS})
  target_link_libraries(segmesh PRIVATE ${WAYLAND_CLIENT_LIBRARIES})
  set(SEGMESH_HAS_LINUX_WINDOW_BACKEND ON)
endif()

if(X11_FOUND)
  message(STATUS "Including X11 support")
  target_compile_definitions(segmesh PRIVATE GLFW_EXPOSE_NATIVE_X11)
  if(X11_INCLUDE_DIR)
    target_include_directories(segmesh PRIVATE ${X11_INCLUDE_DIR})
  endif()
  if(X11_LIBRARIES)
    target_link_libraries(segmesh PRIVATE ${X11_LIBRARIES})
  endif()
  set(SEGMESH_HAS_LINUX_WINDOW_BACKEND ON)
endif()

if(NOT SEGMESH_HAS_LINUX_WINDOW_BACKEND)
  message(FATAL_ERROR
    "No supported Linux window backend development files were found. "
    "Install Wayland and/or X11 development packages before building."
  )
endif()


if(UNIX AND NOT APPLE)
  target_link_libraries(segmesh PRIVATE dl pthread)
endif()
