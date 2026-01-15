/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "gpu/gpu_util.h"
#include "gpu/shading_api.h"

namespace visrtx {

enum class RayType
{
  PRIMARY
};

DECLARE_FRAME_DATA(frameData)

VISRTX_GLOBAL void __anyhit__primary()
{
  ray::cullbackFaces();
}

VISRTX_GLOBAL void __closesthit__primary()
{
  ray::populateHit();
}

VISRTX_GLOBAL void __miss__()
{
  // no-op
}

// Helper function to trace a ray for a specific pixel and return object ID
VISRTX_DEVICE uint32_t traceObjectId(glm::uvec2 pixel)
{
  if (pixel.x >= frameData.fb.size.x || pixel.y >= frameData.fb.size.y)
    return ~0u;

  ScreenSample ss;
  ss.pixel = pixel;
  ss.screen = (glm::vec2(pixel) + 0.5f) * frameData.fb.invSize;
  ss.frameData = &frameData;

  auto ray = makePrimaryRay(ss, true /*pixel centered*/);

  SurfaceHit surfaceHit;
  surfaceHit.foundHit = false;
  intersectSurface(ss,
      ray,
      RayType::PRIMARY,
      &surfaceHit,
      primaryRayOptiXFlags(frameData.renderer));

  return surfaceHit.foundHit ? surfaceHit.objID : ~0u;
}

VISRTX_GLOBAL void __raygen__()
{
  auto &rendererParams = frameData.renderer.params.edges;

  auto ss = createScreenSample(frameData);
  if (pixelOutOfFrame(ss.pixel, frameData.fb))
    return;

  glm::uvec2 pixel = ss.pixel;

  // Sample neighboring pixels and look for object ID discontinuities
  uint32_t objIds[3][3];
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      glm::uvec2 samplePixel = pixel;
      if (pixel.x >= (uint32_t)abs(dx))
        samplePixel.x += dx;
      if (pixel.y >= (uint32_t)abs(dy))
        samplePixel.y += dy;
      objIds[dy + 1][dx + 1] = traceObjectId(samplePixel);
    }
  }

  const uint32_t centerId = objIds[1][1];
  int diffCount = 0;
  for (int dy = 0; dy < 3; ++dy) {
    for (int dx = 0; dx < 3; ++dx) {
      if (dx == 1 && dy == 1)
        continue;
      if (objIds[dy][dx] != centerId)
        diffCount++;
    }
  }

  const float edgeStrength = diffCount / 8.f;
  float edgeIntensity = edgeStrength > rendererParams.threshold ? 1.f : 0.f;

  if (rendererParams.invert) {
    edgeIntensity = 1.f - edgeIntensity;
  }

  vec3 outputColor(edgeIntensity);

  accumResults(frameData,
      pixel,
      vec4(outputColor, 1.f),
      1e30f, // depth
      outputColor,
      vec3(0.f), // normal
      ~0u, // primID
      ~0u, // objID
      ~0u); // instID
}

} // namespace visrtx
