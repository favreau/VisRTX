// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/procedural.hpp"

namespace tsd::io {

void generate_default_lights(Scene &scene)
{
  auto lightsRoot = scene.defaultLayer()->root()->insert_first_child({});
  (*lightsRoot)->name() = "defaultLights";

  // Add a directional light to simulate the sun
  auto sunLight = scene.createObject<tsd::core::Light>(
      tsd::core::tokens::light::directional);
  sunLight->setName("sun_light");
  sunLight->setParameter("direction", tsd::math::float3(-0.5f, -0.5f, 0.75f));
  sunLight->setParameter(
      "color", tsd::math::float3(1.f, 0.95f, 0.8f)); // Warm sunlight color
  sunLight->setParameter("irradiance", 10.f);

  lightsRoot->insert_first_child({sunLight});
}

} // namespace tsd::io
