// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "RenderPass.h"

namespace tsd::rendering {

enum class CosmosMode
{
  NONE,
  DEPTH,
  EDGES
};

struct CosmosVisualizePass : public RenderPass
{
  CosmosVisualizePass();
  ~CosmosVisualizePass() override;

  void setCosmosMode(CosmosMode mode);
  void setDepthRange(float minDepth, float maxDepth);
  void setEdgeThreshold(float threshold);
  void setInvertEdges(bool invert);

 private:
  void render(RenderBuffers &b, int stageId) override;

  CosmosMode m_cosmosMode{CosmosMode::NONE};
  float m_minDepth{0.f};
  float m_maxDepth{1.f};
  float m_edgeThreshold{0.1f};
  bool m_invertEdges{false};
};

} // namespace tsd::rendering
