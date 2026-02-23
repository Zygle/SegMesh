#include <bgfx/bgfx.h>
#include <bgfx/platform.h>
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <wayland-client-core.h>
#include <bx/math.h>
#include "imgui.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
struct Float3
{
    float x;
    float y;
    float z;
};

Float3 add(const Float3& a, const Float3& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Float3 sub(const Float3& a, const Float3& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Float3 cross(const Float3& a, const Float3& b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

Float3 normalize(const Float3& v, const Float3& fallback = {0.0f, 1.0f, 0.0f})
{
    const float lenSq = v.x * v.x + v.y * v.y + v.z * v.z;
    if (lenSq < 1.0e-12f)
    {
        return fallback;
    }

    const float invLen = 1.0f / std::sqrt(lenSq);
    return {v.x * invLen, v.y * invLen, v.z * invLen};
}

uint32_t packNormal(const Float3& normal)
{
    auto toByte = [](float value) -> uint8_t
    {
        const float normalized = std::clamp(value * 0.5f + 0.5f, 0.0f, 1.0f);
        return static_cast<uint8_t>(normalized * 255.0f + 0.5f);
    };

    const uint32_t nx = toByte(normal.x);
    const uint32_t ny = toByte(normal.y);
    const uint32_t nz = toByte(normal.z);
    return nx | (ny << 8) | (nz << 16) | (0xffu << 24);
}

uint32_t packAbgr(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255)
{
    return uint32_t(r)
        | (uint32_t(g) << 8)
        | (uint32_t(b) << 16)
        | (uint32_t(a) << 24);
}

struct MeshVertex
{
    float x;
    float y;
    float z;
    uint32_t abgr;
    uint32_t normal;

    static bgfx::VertexLayout layout;

    static void initLayout()
    {
        layout.begin()
            .add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float)
            .add(bgfx::Attrib::Color0, 4, bgfx::AttribType::Uint8, true, false)
            .add(bgfx::Attrib::Normal, 4, bgfx::AttribType::Uint8, true, true)
            .end();
    }
};

bgfx::VertexLayout MeshVertex::layout;

struct VertexKey
{
    int position = -1;
    int normal = -1;

    bool operator==(const VertexKey& rhs) const
    {
        return position == rhs.position && normal == rhs.normal;
    }
};

struct VertexKeyHash
{
    std::size_t operator()(const VertexKey& key) const
    {
        const std::size_t p = static_cast<std::size_t>(key.position + 1);
        const std::size_t n = static_cast<std::size_t>(key.normal + 2);
        return (p * 73856093u) ^ (n * 19349663u);
    }
};

struct CpuMesh
{
    std::vector<MeshVertex> vertices;
    std::vector<uint32_t> indices;
};

bool parseFaceToken(const std::string& token, int& outPosition, int& outNormal)
{
    outPosition = 0;
    outNormal = 0;

    try
    {
        const std::size_t firstSlash = token.find('/');
        if (firstSlash == std::string::npos)
        {
            outPosition = std::stoi(token);
            return true;
        }

        outPosition = std::stoi(token.substr(0, firstSlash));

        const std::size_t secondSlash = token.find('/', firstSlash + 1);
        if (secondSlash == std::string::npos)
        {
            return true;
        }

        if (secondSlash + 1 < token.size())
        {
            outNormal = std::stoi(token.substr(secondSlash + 1));
        }

        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool resolveObjIndex(int rawIndex, std::size_t count, int& outIndex)
{
    if (rawIndex > 0)
    {
        outIndex = rawIndex - 1;
    }
    else if (rawIndex < 0)
    {
        outIndex = static_cast<int>(count) + rawIndex;
    }
    else
    {
        return false;
    }

    return outIndex >= 0 && outIndex < static_cast<int>(count);
}

bool loadObjMesh(const std::filesystem::path& filePath, CpuMesh& outMesh, std::string& error)
{
    std::ifstream input(filePath);
    if (!input)
    {
        error = "Unable to open OBJ file: " + filePath.string();
        return false;
    }

    std::vector<Float3> positions;
    std::vector<Float3> normals;

    std::unordered_map<VertexKey, uint32_t, VertexKeyHash> vertexMap;
    std::vector<Float3> uniquePositions;
    std::vector<Float3> accumulatedNormals;
    std::vector<bool> hasImportedNormal;

    auto makeVertex = [&](int positionIndex, int normalIndex) -> uint32_t
    {
        const VertexKey key{positionIndex, normalIndex};
        const auto it = vertexMap.find(key);
        if (it != vertexMap.end())
        {
            return it->second;
        }

        const Float3& p = positions[static_cast<std::size_t>(positionIndex)];
        const bool hasNormal = normalIndex >= 0;
        const Float3 normal = hasNormal
            ? normalize(normals[static_cast<std::size_t>(normalIndex)])
            : Float3{0.0f, 1.0f, 0.0f};

        MeshVertex v{};
        v.x = p.x;
        v.y = p.y;
        v.z = p.z;
        v.abgr = packAbgr(210, 210, 210);
        v.normal = packNormal(normal);

        const uint32_t newIndex = static_cast<uint32_t>(outMesh.vertices.size());
        outMesh.vertices.push_back(v);
        uniquePositions.push_back(p);
        accumulatedNormals.push_back({0.0f, 0.0f, 0.0f});
        hasImportedNormal.push_back(hasNormal);
        vertexMap.emplace(key, newIndex);
        return newIndex;
    };

    std::string line;
    uint32_t lineNumber = 0;
    while (std::getline(input, line))
    {
        ++lineNumber;

        const std::size_t commentPos = line.find('#');
        if (commentPos != std::string::npos)
        {
            line.erase(commentPos);
        }

        std::istringstream iss(line);
        std::string type;
        iss >> type;
        if (type.empty())
        {
            continue;
        }

        if (type == "v")
        {
            Float3 p{};
            if (!(iss >> p.x >> p.y >> p.z))
            {
                error = "Malformed vertex at line " + std::to_string(lineNumber);
                return false;
            }
            positions.push_back(p);
            continue;
        }

        if (type == "vn")
        {
            Float3 n{};
            if (!(iss >> n.x >> n.y >> n.z))
            {
                error = "Malformed normal at line " + std::to_string(lineNumber);
                return false;
            }
            normals.push_back(n);
            continue;
        }

        if (type != "f")
        {
            continue;
        }

        std::vector<std::pair<int, int> > face;
        std::string token;
        while (iss >> token)
        {
            int rawPosition = 0;
            int rawNormal = 0;
            if (!parseFaceToken(token, rawPosition, rawNormal))
            {
                error = "Malformed face token at line " + std::to_string(lineNumber) + ": " + token;
                return false;
            }

            int positionIndex = -1;
            if (!resolveObjIndex(rawPosition, positions.size(), positionIndex))
            {
                error = "Face position index out of range at line " + std::to_string(lineNumber);
                return false;
            }

            int normalIndex = -1;
            if (rawNormal != 0 && !resolveObjIndex(rawNormal, normals.size(), normalIndex))
            {
                error = "Face normal index out of range at line " + std::to_string(lineNumber);
                return false;
            }

            face.emplace_back(positionIndex, normalIndex);
        }

        if (face.size() < 3)
        {
            continue;
        }

        for (std::size_t i = 1; i + 1 < face.size(); ++i)
        {
            const uint32_t i0 = makeVertex(face[0].first, face[0].second);
            const uint32_t i1 = makeVertex(face[i].first, face[i].second);
            const uint32_t i2 = makeVertex(face[i + 1].first, face[i + 1].second);

            outMesh.indices.push_back(i0);
            outMesh.indices.push_back(i1);
            outMesh.indices.push_back(i2);

            const Float3& p0 = uniquePositions[static_cast<std::size_t>(i0)];
            const Float3& p1 = uniquePositions[static_cast<std::size_t>(i1)];
            const Float3& p2 = uniquePositions[static_cast<std::size_t>(i2)];
            const Float3 faceNormal = normalize(cross(sub(p1, p0), sub(p2, p0)));

            if (!hasImportedNormal[static_cast<std::size_t>(i0)])
            {
                accumulatedNormals[static_cast<std::size_t>(i0)] = add(accumulatedNormals[static_cast<std::size_t>(i0)], faceNormal);
            }
            if (!hasImportedNormal[static_cast<std::size_t>(i1)])
            {
                accumulatedNormals[static_cast<std::size_t>(i1)] = add(accumulatedNormals[static_cast<std::size_t>(i1)], faceNormal);
            }
            if (!hasImportedNormal[static_cast<std::size_t>(i2)])
            {
                accumulatedNormals[static_cast<std::size_t>(i2)] = add(accumulatedNormals[static_cast<std::size_t>(i2)], faceNormal);
            }
        }
    }

    if (outMesh.vertices.empty() || outMesh.indices.empty())
    {
        error = "OBJ has no renderable geometry: " + filePath.string();
        return false;
    }

    for (std::size_t i = 0; i < outMesh.vertices.size(); ++i)
    {
        if (!hasImportedNormal[i])
        {
            const Float3 n = normalize(accumulatedNormals[i]);
            outMesh.vertices[i].normal = packNormal(n);
        }
    }

    Float3 minPos{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
    };
    Float3 maxPos{
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
    };

    for (const MeshVertex& v : outMesh.vertices)
    {
        minPos.x = std::min(minPos.x, v.x);
        minPos.y = std::min(minPos.y, v.y);
        minPos.z = std::min(minPos.z, v.z);

        maxPos.x = std::max(maxPos.x, v.x);
        maxPos.y = std::max(maxPos.y, v.y);
        maxPos.z = std::max(maxPos.z, v.z);
    }

    const Float3 center{
        (minPos.x + maxPos.x) * 0.5f,
        (minPos.y + maxPos.y) * 0.5f,
        (minPos.z + maxPos.z) * 0.5f,
    };

    const float maxExtent = std::max({maxPos.x - minPos.x, maxPos.y - minPos.y, maxPos.z - minPos.z});
    const float scale = maxExtent > 1.0e-6f ? (2.0f / maxExtent) : 1.0f;

    for (MeshVertex& v : outMesh.vertices)
    {
        v.x = (v.x - center.x) * scale;
        v.y = (v.y - center.y) * scale;
        v.z = (v.z - center.z) * scale;
    }

    return true;
}

std::optional<std::filesystem::path> findFirstExisting(const std::vector<std::filesystem::path>& candidates)
{
    for (const std::filesystem::path& path : candidates)
    {
        if (std::filesystem::exists(path))
        {
            return path;
        }
    }

    return std::nullopt;
}

std::vector<std::filesystem::path> findObjFiles(const std::filesystem::path& directory)
{
    std::vector<std::filesystem::path> result;

    std::error_code ec;
    std::filesystem::directory_iterator it(directory, ec);
    if (ec)
    {
        return result;
    }

    const std::filesystem::directory_iterator end;
    for (; it != end; it.increment(ec))
    {
        if (ec)
        {
            break;
        }

        if (!it->is_regular_file(ec))
        {
            continue;
        }

        std::string extension = it->path().extension().string();
        std::transform(
            extension.begin(),
            extension.end(),
            extension.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); }
        );

        if (extension == ".obj")
        {
            result.push_back(it->path());
        }
    }

    std::sort(result.begin(), result.end());
    return result;
}

int findModelIndex(const std::vector<std::filesystem::path>& models, const std::filesystem::path& target)
{
    for (std::size_t i = 0; i < models.size(); ++i)
    {
        std::error_code ec;
        if (std::filesystem::equivalent(models[i], target, ec) && !ec)
        {
            return static_cast<int>(i);
        }
    }

    return -1;
}

std::optional<std::vector<uint8_t> > readBinaryFile(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
    {
        return std::nullopt;
    }

    const std::streamsize size = input.tellg();
    if (size <= 0)
    {
        return std::nullopt;
    }

    std::vector<uint8_t> bytes(static_cast<std::size_t>(size));
    input.seekg(0, std::ios::beg);
    if (!input.read(reinterpret_cast<char*>(bytes.data()), size))
    {
        return std::nullopt;
    }

    return bytes;
}

bgfx::ShaderHandle loadShaderFromFile(const std::filesystem::path& path)
{
    const auto bytes = readBinaryFile(path);
    if (!bytes.has_value())
    {
        return BGFX_INVALID_HANDLE;
    }

    const bgfx::Memory* memory = bgfx::copy(bytes->data(), static_cast<uint32_t>(bytes->size()));
    bgfx::ShaderHandle shader = bgfx::createShader(memory);
    if (bgfx::isValid(shader))
    {
        const std::string name = path.filename().string();
        bgfx::setName(shader, name.c_str());
    }

    return shader;
}

struct GpuMesh
{
    bgfx::VertexBufferHandle vbh = BGFX_INVALID_HANDLE;
    bgfx::IndexBufferHandle ibh = BGFX_INVALID_HANDLE;
    uint32_t indexCount = 0;
};

void destroyGpuMesh(GpuMesh& gpuMesh)
{
    if (bgfx::isValid(gpuMesh.vbh))
    {
        bgfx::destroy(gpuMesh.vbh);
    }

    if (bgfx::isValid(gpuMesh.ibh))
    {
        bgfx::destroy(gpuMesh.ibh);
    }

    gpuMesh.vbh = BGFX_INVALID_HANDLE;
    gpuMesh.ibh = BGFX_INVALID_HANDLE;
    gpuMesh.indexCount = 0;
}

bool uploadMesh(const CpuMesh& mesh, GpuMesh& gpuMesh, std::string& error)
{
    const bgfx::Memory* vmem = bgfx::copy(
        mesh.vertices.data(),
        static_cast<uint32_t>(mesh.vertices.size() * sizeof(MeshVertex))
    );
    gpuMesh.vbh = bgfx::createVertexBuffer(vmem, MeshVertex::layout);

    const bgfx::Memory* imem = bgfx::copy(
        mesh.indices.data(),
        static_cast<uint32_t>(mesh.indices.size() * sizeof(uint32_t))
    );
    gpuMesh.ibh = bgfx::createIndexBuffer(imem, BGFX_BUFFER_INDEX32);
    gpuMesh.indexCount = static_cast<uint32_t>(mesh.indices.size());

    if (!bgfx::isValid(gpuMesh.vbh) || !bgfx::isValid(gpuMesh.ibh))
    {
        error = "Failed to create bgfx mesh buffers.";
        destroyGpuMesh(gpuMesh);
        return false;
    }

    return true;
}

double g_scrollDelta = 0.0;
int g_inputChar = -1;

void onScroll(GLFWwindow*, double, double yoffset)
{
    g_scrollDelta += yoffset;
}

void onChar(GLFWwindow*, unsigned int codepoint)
{
    g_inputChar = static_cast<int>(codepoint);
}
} // namespace

int main(int argc, char** argv)
{
    std::vector<std::filesystem::path> modelPaths;
    const auto modelDir = findFirstExisting({
        "assets/shaders",
        "../assets/shaders",
        "../../assets/shaders",
    });
    if (modelDir.has_value())
    {
        modelPaths = findObjFiles(*modelDir);
    }

    int selectedModelIndex = 0;
    if (argc > 1)
    {
        const std::filesystem::path cliPath(argv[1]);
        if (!std::filesystem::exists(cliPath))
        {
            std::cerr << "OBJ file not found: " << cliPath << "\n";
            return 1;
        }

        const int existingIndex = findModelIndex(modelPaths, cliPath);
        if (existingIndex >= 0)
        {
            selectedModelIndex = existingIndex;
        }
        else
        {
            modelPaths.insert(modelPaths.begin(), cliPath);
            selectedModelIndex = 0;
        }
    }

    if (modelPaths.empty())
    {
        const auto defaultObj = findFirstExisting({
            "assets/model.obj",
            "submods/bgfx/examples/assets/meshes/bunny.obj",
            "../submods/bgfx/examples/assets/meshes/bunny.obj",
        });

        if (!defaultObj.has_value())
        {
            std::cerr << "No OBJ file found. Add .obj files to assets/shaders or pass one on the command line.\n";
            return 1;
        }

        modelPaths.push_back(*defaultObj);
        selectedModelIndex = 0;
    }

    std::filesystem::path objPath = modelPaths[static_cast<std::size_t>(selectedModelIndex)];

    CpuMesh mesh{};
    std::string error;
    if (!loadObjMesh(objPath, mesh, error))
    {
        std::cerr << error << "\n";
        return 1;
    }

    if (!glfwInit())
    {
        std::cerr << "Failed to init GLFW\n";
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(1280, 720, "SegMesh - OBJ Renderer", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return 1;
    }

    glfwSetScrollCallback(window, onScroll);
    glfwSetCharCallback(window, onChar);

    int width = 0;
    int height = 0;
    while (width == 0 || height == 0)
    {
        glfwWaitEvents();
        glfwGetFramebufferSize(window, &width, &height);
    }

    wl_display* display = glfwGetWaylandDisplay();
    wl_surface* surface = glfwGetWaylandWindow(window);

    if (!display || !surface)
    {
        std::cerr << "Wayland backend is required by this sample.\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    bgfx::renderFrame();

    bgfx::PlatformData platformData{};
    platformData.ndt = display;
    platformData.nwh = surface;
    platformData.context = nullptr;
    platformData.backBuffer = nullptr;
    platformData.backBufferDS = nullptr;
    platformData.type = bgfx::NativeWindowHandleType::Wayland;

    bgfx::Init init{};
    init.type = bgfx::RendererType::Vulkan;
    init.platformData = platformData;
    init.resolution.width = static_cast<uint32_t>(width);
    init.resolution.height = static_cast<uint32_t>(height);
    init.resolution.reset = BGFX_RESET_VSYNC;

    if (!bgfx::init(init))
    {
        std::cerr << "bgfx::init failed\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    const auto shaderDir = findFirstExisting({
        "submods/bgfx/examples/runtime/shaders/spirv",
        "../submods/bgfx/examples/runtime/shaders/spirv",
        "../../submods/bgfx/examples/runtime/shaders/spirv",
    });
    if (!shaderDir.has_value())
    {
        std::cerr << "Unable to find bgfx runtime shader directory.\n";
        bgfx::shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    bgfx::ShaderHandle vsh = loadShaderFromFile(*shaderDir / "vs_cubes.bin");
    bgfx::ShaderHandle fsh = loadShaderFromFile(*shaderDir / "fs_cubes.bin");
    if (!bgfx::isValid(vsh) || !bgfx::isValid(fsh))
    {
        std::cerr << "Failed to load shader binaries from: " << *shaderDir << "\n";
        if (bgfx::isValid(vsh))
        {
            bgfx::destroy(vsh);
        }
        if (bgfx::isValid(fsh))
        {
            bgfx::destroy(fsh);
        }
        bgfx::shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    bgfx::ProgramHandle meshProgram = bgfx::createProgram(vsh, fsh, true);
    if (!bgfx::isValid(meshProgram))
    {
        std::cerr << "Failed to create mesh program.\n";
        bgfx::shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    MeshVertex::initLayout();
    GpuMesh gpuMesh{};
    if (!uploadMesh(mesh, gpuMesh, error))
    {
        std::cerr << error << "\n";
        if (bgfx::isValid(meshProgram))
        {
            bgfx::destroy(meshProgram);
        }
        bgfx::shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    imguiCreate(18.0f);

    float cameraDistance = 3.0f;
    float cameraYaw = 0.0f;
    float cameraPitch = 0.35f;
    float modelRotation = 0.0f;
    float rotateSpeed = 0.8f;
    bool autoRotate = true;
    bool wireframe = false;

    std::string modelLoadError;

    float previousTime = static_cast<float>(glfwGetTime());
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        int newWidth = 0;
        int newHeight = 0;
        glfwGetFramebufferSize(window, &newWidth, &newHeight);
        if (newWidth != width || newHeight != height)
        {
            width = newWidth;
            height = newHeight;
            if (width > 0 && height > 0)
            {
                bgfx::reset(static_cast<uint32_t>(width), static_cast<uint32_t>(height), BGFX_RESET_VSYNC);
            }
        }

        if (width == 0 || height == 0)
        {
            continue;
        }

        const float now = static_cast<float>(glfwGetTime());
        const float dt = now - previousTime;
        previousTime = now;
        if (autoRotate)
        {
            modelRotation += rotateSpeed * dt;
        }

        const bgfx::Caps* caps = bgfx::getCaps();

        bgfx::setViewClear(0, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, 0x303040ff, 1.0f, 0);
        bgfx::setViewRect(0, 0, 0, static_cast<uint16_t>(width), static_cast<uint16_t>(height));

        const float eyeX = std::sin(cameraYaw) * std::cos(cameraPitch) * cameraDistance;
        const float eyeY = std::sin(cameraPitch) * cameraDistance;
        const float eyeZ = std::cos(cameraYaw) * std::cos(cameraPitch) * cameraDistance;

        const bx::Vec3 eye{eyeX, eyeY, eyeZ};
        const bx::Vec3 at{0.0f, 0.0f, 0.0f};
        float view[16];
        bx::mtxLookAt(view, eye, at);

        float proj[16];
        bx::mtxProj(proj, 60.0f, static_cast<float>(width) / static_cast<float>(height), 0.01f, 100.0f, caps->homogeneousDepth);
        bgfx::setViewTransform(0, view, proj);

        float model[16];
        bx::mtxRotateY(model, modelRotation);
        bgfx::setTransform(model);
        bgfx::setVertexBuffer(0, gpuMesh.vbh);
        bgfx::setIndexBuffer(gpuMesh.ibh, 0, gpuMesh.indexCount);

        uint64_t state = BGFX_STATE_WRITE_RGB
            | BGFX_STATE_WRITE_A
            | BGFX_STATE_WRITE_Z
            | BGFX_STATE_DEPTH_TEST_LESS
            | BGFX_STATE_MSAA;
        if (wireframe)
        {
            state |= BGFX_STATE_PT_LINES;
        }
        bgfx::setState(state);
        bgfx::submit(0, meshProgram);

        double mouseX = 0.0;
        double mouseY = 0.0;
        glfwGetCursorPos(window, &mouseX, &mouseY);
        uint8_t mouseButtons = 0;
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
        {
            mouseButtons |= IMGUI_MBUT_LEFT;
        }
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
        {
            mouseButtons |= IMGUI_MBUT_RIGHT;
        }
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS)
        {
            mouseButtons |= IMGUI_MBUT_MIDDLE;
        }

        const int32_t scroll = static_cast<int32_t>(std::lround(g_scrollDelta));
        g_scrollDelta = 0.0;
        const int inputChar = g_inputChar;
        g_inputChar = -1;

        imguiBeginFrame(
            static_cast<int32_t>(mouseX),
            static_cast<int32_t>(mouseY),
            mouseButtons,
            scroll,
            static_cast<uint16_t>(width),
            static_cast<uint16_t>(height),
            inputChar,
            1
        );

        ImGui::Begin("Renderer");
        const std::string activeModelLabel = objPath.filename().string();
        int pendingModelIndex = selectedModelIndex;
        if (ImGui::BeginCombo("Model", activeModelLabel.c_str()))
        {
            for (int i = 0; i < static_cast<int>(modelPaths.size()); ++i)
            {
                const bool selected = i == selectedModelIndex;
                const std::string itemLabel = modelPaths[static_cast<std::size_t>(i)].filename().string();
                if (ImGui::Selectable(itemLabel.c_str(), selected))
                {
                    pendingModelIndex = i;
                }

                if (selected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        if (pendingModelIndex != selectedModelIndex)
        {
            CpuMesh newMesh{};
            GpuMesh newGpuMesh{};
            std::string loadError;
            const std::filesystem::path newPath = modelPaths[static_cast<std::size_t>(pendingModelIndex)];
            if (loadObjMesh(newPath, newMesh, loadError) && uploadMesh(newMesh, newGpuMesh, loadError))
            {
                destroyGpuMesh(gpuMesh);
                mesh = std::move(newMesh);
                gpuMesh = newGpuMesh;
                objPath = newPath;
                selectedModelIndex = pendingModelIndex;
                modelRotation = 0.0f;
                modelLoadError.clear();
            }
            else
            {
                destroyGpuMesh(newGpuMesh);
                modelLoadError = loadError;
            }
        }

        ImGui::Text("Renderer: %s", bgfx::getRendererName(bgfx::getRendererType()));
        ImGui::Text("OBJ: %s", objPath.string().c_str());
        ImGui::Text("Available models: %u", static_cast<uint32_t>(modelPaths.size()));
        ImGui::Text("Vertices: %u", static_cast<uint32_t>(mesh.vertices.size()));
        ImGui::Text("Triangles: %u", static_cast<uint32_t>(mesh.indices.size() / 3));
        if (!modelLoadError.empty())
        {
            ImGui::Separator();
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Model load error: %s", modelLoadError.c_str());
        }
        ImGui::Separator();
        ImGui::Checkbox("Auto rotate", &autoRotate);
        ImGui::SliderFloat("Rotate speed", &rotateSpeed, 0.0f, 3.0f);
        ImGui::SliderFloat("Camera distance", &cameraDistance, 1.0f, 10.0f);
        ImGui::SliderFloat("Camera yaw", &cameraYaw, -3.14f, 3.14f);
        ImGui::SliderFloat("Camera pitch", &cameraPitch, -1.2f, 1.2f);
        ImGui::Checkbox("Wireframe", &wireframe);
        ImGui::End();

        imguiEndFrame();

        bgfx::frame();
    }

    imguiDestroy();

    destroyGpuMesh(gpuMesh);
    if (bgfx::isValid(meshProgram))
    {
        bgfx::destroy(meshProgram);
    }

    bgfx::shutdown();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
