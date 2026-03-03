#include "mesh_types.h"

namespace segmesh
{
bgfx::VertexLayout MeshVertex::layout;

void MeshVertex::initLayout()
{
    layout.begin()
        .add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float)
        .add(bgfx::Attrib::Normal, 4, bgfx::AttribType::Uint8, true, false)
        .end();
}
}
