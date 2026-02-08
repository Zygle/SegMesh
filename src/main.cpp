#include <bgfx/bgfx.h>
#include <bgfx/platform.h>

#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WAYLAND
#include <GLFW/glfw3native.h>

#include <wayland-client-core.h>

#include <cstdint>
#include <cmath>
#include <iostream>

int main()
{
    if (!glfwInit())
    {
        std::cerr << "Failed to init GLFW\n";
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(1280, 720, "SegMesh (bgfx)", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return 1;
    }

    int w = 0, h = 0;
    // Wayland can report 0x0 initially; wait for a real size.
    while (w == 0 || h == 0)
    {
        glfwWaitEvents();
        glfwGetFramebufferSize(window, &w, &h);
    }

    wl_display* display = glfwGetWaylandDisplay();
    wl_surface* surface = glfwGetWaylandWindow(window);

    if (!display || !surface)
    {
        std::cerr << "ERROR: Not running on Wayland backend.\n";
        return 1;
    }

    bgfx::renderFrame();

    bgfx::PlatformData pd{};
    pd.ndt = display;
    pd.nwh = surface;
    pd.context = nullptr;
    pd.backBuffer = nullptr;
    pd.backBufferDS = nullptr;
    pd.type = bgfx::NativeWindowHandleType::Wayland;

    bgfx::Init init{};
    init.type = bgfx::RendererType::Vulkan;
    init.platformData = pd;
    init.resolution.width  = (uint32_t)w;
    init.resolution.height = (uint32_t)h;
    init.resolution.reset  = BGFX_RESET_VSYNC;
    init.debug = BGFX_DEBUG_TEXT;

    if (!bgfx::init(init))
    {
        std::cerr << "bgfx::init failed\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    std::cerr << "bgfx renderer: " << bgfx::getRendererName(bgfx::getRendererType()) << "\n";

    bgfx::setDebug(BGFX_DEBUG_TEXT);

    bgfx::reset((uint32_t)w, (uint32_t)h, BGFX_RESET_VSYNC);

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        int newW = 0, newH = 0;
        glfwGetFramebufferSize(window, &newW, &newH);
        if (newW != w || newH != h)
        {
            w = newW; h = newH;
            if (w == 0 || h == 0)
            {
                continue;
            }

            bgfx::reset((uint32_t)w, (uint32_t)h, BGFX_RESET_VSYNC);
        }

        const float t = static_cast<float>(glfwGetTime());
        const uint8_t r = static_cast<uint8_t>(127.0f + 127.0f * std::sin(t * 1.1f));
        const uint8_t g = static_cast<uint8_t>(127.0f + 127.0f * std::sin(t * 1.7f + 1.0f));
        const uint8_t b = static_cast<uint8_t>(127.0f + 127.0f * std::sin(t * 2.3f + 2.0f));
        const uint32_t abgr = 0xff000000u | (uint32_t(b) << 16) | (uint32_t(g) << 8) | uint32_t(r);

        bgfx::setViewClear(0, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, abgr, 1.0f, 0);
        bgfx::setViewRect(0, 0, 0, (uint16_t)w, (uint16_t)h);

        bgfx::touch(0);

        bgfx::dbgTextClear();
        bgfx::dbgTextPrintf(0, 0, 0x0f, "SegMesh: bgfx %s", bgfx::getRendererName(bgfx::getRendererType()));
        bgfx::dbgTextPrintf(0, 1, 0x0f, "Animating clear color.");

        bgfx::frame();
    }

    bgfx::shutdown();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}


int _main_(int _argc, char** _argv) {return 0;};
