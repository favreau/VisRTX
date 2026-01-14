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

#include "Edges.h"
// ptx
#include "Edges_ptx.h"

namespace visrtx {

static const std::vector<HitgroupFunctionNames> g_edgesHitNames = {
    {"__closesthit__primary", "__anyhit__primary"}};

static const std::vector<std::string> g_edgesMissNames = {"__miss__"};

Edges::Edges(DeviceGlobalState *s) : Renderer(s) {}

void Edges::commitParameters()
{
  Renderer::commitParameters();
  m_threshold = getParam<float>("threshold", 0.01f);
  m_invert = getParam<bool>("invert", false);
  m_sampleLimit = 1; // single-shot renderer
  m_denoise = false; // never denoise
}

void Edges::populateFrameData(FrameGPUData &fd) const
{
  Renderer::populateFrameData(fd);
  fd.renderer.params.edges.threshold = m_threshold;
  fd.renderer.params.edges.invert = m_invert;
}

OptixModule Edges::optixModule() const
{
  return deviceState()->rendererModules.edges;
}

Span<HitgroupFunctionNames> Edges::hitgroupSbtNames() const
{
  return make_Span(g_edgesHitNames.data(), g_edgesHitNames.size());
}

Span<std::string> Edges::missSbtNames() const
{
  return make_Span(g_edgesMissNames.data(), g_edgesMissNames.size());
}

ptx_blob Edges::ptx()
{
  return {Edges_ptx, sizeof(Edges_ptx)};
}

} // namespace visrtx
