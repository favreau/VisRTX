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

VISRTX_GLOBAL void __raygen__()
{
  auto &rendererParams = frameData.renderer.params.depth;

  auto ss = createScreenSample(frameData);
  if (pixelOutOfFrame(ss.pixel, frameData.fb))
    return;

  auto ray = makePrimaryRay(ss, true /*pixel centered*/);

  // Trace ray
  SurfaceHit surfaceHit;
  surfaceHit.foundHit = false;
  intersectSurface(ss,
      ray,
      RayType::PRIMARY,
      &surfaceHit,
      primaryRayOptiXFlags(frameData.renderer));

  vec3 outputColor(1.f); // default white (far)

  if (surfaceHit.foundHit) {
    // Normalize depth to [0,1] range
    float depth = surfaceHit.t;
    float normalizedDepth = (depth - rendererParams.minDepth)
        / (rendererParams.maxDepth - rendererParams.minDepth);
    normalizedDepth = glm::clamp(normalizedDepth, 0.f, 1.f);

    // Grayscale visualization (near=black, far=white)
    outputColor = vec3(normalizedDepth);
  }

  accumResults(frameData,
      ss.pixel,
      vec4(outputColor, 1.f),
      surfaceHit.foundHit ? surfaceHit.t : 1e30f,
      outputColor,
      vec3(0.f), // normal
      ~0u, // primID
      ~0u, // objID
      ~0u); // instID
}

} // namespace visrtx
