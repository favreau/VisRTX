// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "CosmosVisualizePass.h"
// std
#include <algorithm>
#include <cmath>

#include "detail/parallel_for.h"
#include "helium/helium_math.h"

namespace tsd::rendering {

// Cosmos shader kernels //////////////////////////////////////////////////////

void computeCosmosDepthImage(
    RenderBuffers &b, float minDepth, float maxDepth, tsd::math::uint2 size)
{
  detail::parallel_for(
      b.stream, 0u, uint32_t(size.x * size.y), [=] DEVICE_FCN(uint32_t i) {
        const float depth = b.depth[i];
        const float range = maxDepth - minDepth;
        const float v = range > 0.f
            ? std::clamp((depth - minDepth) / range, 0.f, 1.f)
            : 0.f;
        b.color[i] = helium::cvt_color_to_uint32({tsd::math::float3(v), 1.f});
      });
}

void computeCosmosEdgesImage(
    RenderBuffers &b, float threshold, bool invert, tsd::math::uint2 size)
{
  detail::parallel_for(
      b.stream, 0u, uint32_t(size.x * size.y), [=] DEVICE_FCN(uint32_t i) {
        const uint32_t x = i % size.x;
        const uint32_t y = i / size.x;

        // Edge detection using Sobel operator on depth buffer
        float edge = 0.f;

        if (x > 0 && x < size.x - 1 && y > 0 && y < size.y - 1) {
          const uint32_t idx = y * size.x + x;

          // Sobel kernels for edge detection
          const float d00 = b.depth[idx - size.x - 1];
          const float d01 = b.depth[idx - size.x];
          const float d02 = b.depth[idx - size.x + 1];
          const float d10 = b.depth[idx - 1];
          const float d12 = b.depth[idx + 1];
          const float d20 = b.depth[idx + size.x - 1];
          const float d21 = b.depth[idx + size.x];
          const float d22 = b.depth[idx + size.x + 1];

          // Sobel X and Y gradients
          const float gx = -d00 + d02 - 2.f * d10 + 2.f * d12 - d20 + d22;
          const float gy = -d00 - 2.f * d01 - d02 + d20 + 2.f * d21 + d22;

          // Gradient magnitude
          edge = std::sqrt(gx * gx + gy * gy);

          // Normalize and threshold
          edge = edge > threshold ? 1.f : 0.f;
        }

        // Invert if requested (black edges on white, or white edges on black)
        if (invert) {
          edge = 1.f - edge;
        }

        b.color[i] =
            helium::cvt_color_to_uint32({tsd::math::float3(edge), 1.f});
      });
}

// CosmosVisualizePass definitions ////////////////////////////////////////////

CosmosVisualizePass::CosmosVisualizePass() = default;

CosmosVisualizePass::~CosmosVisualizePass() = default;

void CosmosVisualizePass::setCosmosMode(CosmosMode mode)
{
  m_cosmosMode = mode;
  setEnabled(mode != CosmosMode::NONE);
}

void CosmosVisualizePass::setDepthRange(float minDepth, float maxDepth)
{
  m_minDepth = minDepth;
  m_maxDepth = maxDepth;
}

void CosmosVisualizePass::setEdgeThreshold(float threshold)
{
  m_edgeThreshold = threshold;
}

void CosmosVisualizePass::setInvertEdges(bool invert)
{
  m_invertEdges = invert;
}

void CosmosVisualizePass::render(RenderBuffers &b, int stageId)
{
  if (stageId == 0 || m_cosmosMode == CosmosMode::NONE)
    return;

  const auto size = getDimensions();

  switch (m_cosmosMode) {
  case CosmosMode::DEPTH:
    computeCosmosDepthImage(b, m_minDepth, m_maxDepth, size);
    break;
  case CosmosMode::EDGES:
    computeCosmosEdgesImage(b, m_edgeThreshold, m_invertEdges, size);
    break;
  default:
    break;
  }
}

} // namespace tsd::rendering
