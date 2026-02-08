#include <bgfx/bgfx.h>
#include <bgfx/platform.h>

#include <GLFW/glfw3.h>

// Enable native handle getters:
#define GLFW_EXPOSE_NATIVE_WAYLAND
#define GLFW_EXPOSE_NATIVE_X11
#include <GLFW/glfw3native.h>

#include <cstdint>
#include <iostream>

static void setBgfxPlatformData(GLFWwindow* window)
{
    bgfx::PlatformData pd{};
    pd.ndt = nullptr;
    pd.nwh = nullptr;

#if GLFW_VERSION_MAJOR > 3 || (GLFW_VERSION_MAJOR == 3 && GLFW_VERSION_MINOR >= 4)
    const int platform = glfwGetPlatform();
    if (platform == GLFW_PLATFORM_WAYLAND)
    {
        pd.ndt = glfwGetWaylandDisplay();
        pd.nwh = (void*)glfwGetWaylandWindow(window); // wl_surface*
    }
    else if (platform == GLFW_PLATFORM_X11)
    {
        pd.ndt = glfwGetX11Display();
        pd.nwh = (void*)(uintptr_t)glfwGetX11Window(window); // Window
    }
    else
    {
        std::cerr << "Unsupported GLFW platform backend.\n";
        std::exit(1);
    }
#else
    // GLFW < 3.4: assume Wayland if available at runtime.
    pd.ndt = glfwGetWaylandDisplay();
    pd.nwh = (void*)glfwGetWaylandWindow(window);
#endif

    bgfx::setPlatformData(pd);
}

int main()
{
    if (!glfwInit())
    {
        std::cerr << "Failed to init GLFW\n";
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "SegMesh", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return 1;
    }

    setBgfxPlatformData(window);

    int w = 0, h = 0;
    glfwGetFramebufferSize(window, &w, &h);

    bgfx::Init init{};
    init.type = bgfx::RendererType::Vulkan;
    init.resolution.width  = (uint32_t)w;
    init.resolution.height = (uint32_t)h;
    init.resolution.reset  = BGFX_RESET_VSYNC;

    if (!bgfx::init(init))
    {
        std::cerr << "bgfx::init failed\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    bgfx::setViewClear(
        0,
        BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH,
        0x1f2630ff,
        1.0f,
        0
    );
    bgfx::setViewRect(0, 0, 0, (uint16_t)w, (uint16_t)h);

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        int newW = 0, newH = 0;
        glfwGetFramebufferSize(window, &newW, &newH);
        if (newW != w || newH != h)
        {
            w = newW; h = newH;
            bgfx::reset((uint32_t)w, (uint32_t)h, BGFX_RESET_VSYNC);
            bgfx::setViewRect(0, 0, 0, (uint16_t)w, (uint16_t)h);
        }

        // Ensure view 0 is “touched” so clearing happens even with no draw calls.
        bgfx::touch(0);

        bgfx::frame();
    }

    bgfx::shutdown();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
