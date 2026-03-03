#pragma once

#include "mesh_types.h"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace segmesh
{
bool loadObjMesh(const std::filesystem::path& filePath, CpuMesh& outMesh, std::string& error);

std::optional<std::filesystem::path> findFirstExisting(const std::vector<std::filesystem::path>& candidates);
std::vector<std::filesystem::path> findObjFiles(const std::filesystem::path& directory);
int findModelIndex(const std::vector<std::filesystem::path>& models, const std::filesystem::path& target);
}
