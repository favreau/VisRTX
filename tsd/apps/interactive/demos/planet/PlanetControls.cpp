// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "PlanetControls.h"
// tsd_core
#include <tsd/core/Logging.hpp>
#include <tsd/core/scene/objects/Light.hpp>
#include <tsd/core/scene/objects/SpatialField.hpp>
// tsd_io
#include <tsd/io/importers.hpp>
// tsd_ui_imgui
#include <tsd/ui/imgui/Application.h>
#include <tsd/ui/imgui/modals/BlockingTaskModal.h>
// cuda
#include <cuda_runtime.h>

namespace tsd::demo {

PlanetControls::PlanetControls(
    tsd::ui::imgui::Application *app, const char *name)
    : tsd::ui::imgui::Window(app, name)
{}

void PlanetControls::buildUI()
{
  updateTimeStepFromAnimation();

  ImGui::Text("Planet Data");
  ImGui::Separator();

  if (m_cloudsField)
    ImGui::Text("Clouds: %d timesteps", m_maxTimeSteps);
  else
    ImGui::TextDisabled("Clouds: not loaded");

  if (m_auroraField)
    ImGui::Text("Aurora: active");
  else
    ImGui::TextDisabled("Aurora: not loaded");

  if (m_magneticField)
    ImGui::Text("Magnetic: active");
  else
    ImGui::TextDisabled("Magnetic: not loaded");

  if (m_planetField)
    ImGui::Text("Planet: active");
  else
    ImGui::TextDisabled("Planet: not loaded");

  ImGui::Spacing();
  ImGui::Text("Use the Animation panel to control time");
}

void PlanetControls::updateTimeStepFromAnimation()
{
  if (!m_cloudsField && !m_auroraField)
    return;

  auto *core = appCore();
  auto &scene = core->tsd.scene;

  // Get current animation time (0.0 to 1.0)
  float animTime = scene.getAnimationTime();

  // Only update if time changed
  if (animTime == m_lastAnimationTime)
    return;

  m_lastAnimationTime = animTime;

  // Convert normalized time to timestep
  int currentTimeStep = m_maxTimeSteps > 1
      ? static_cast<int>(animTime * (m_maxTimeSteps - 1))
      : 0;
  currentTimeStep = std::clamp(currentTimeStep, 0, m_maxTimeSteps - 1);

  // Update clouds data
  if (m_cloudsField) {
    auto filepathParam = m_cloudsField->getMetadataValue("filepath");
    if (filepathParam.valid()) {
      std::string cloudsFile = filepathParam.getString();
      if (tsd::io::update_CLOUDS(
              scene, m_cloudsField, cloudsFile.c_str(), currentTimeStep)) {
        scene.signalLayerChange(scene.defaultLayer());
      }
    }
  }

  // Update aurora time parameter
  if (m_auroraField) {
    m_auroraField->setParameter("time", animTime * 100.f);
  }
}

void PlanetControls::setupAnimations()
{
  auto *core = appCore();

  // Register callback with Core for animation time changes
  core->animationTimeChangedCallback = [this]() {
    // Force update by resetting last time
    m_lastAnimationTime = -1.f;
    updateTimeStepFromAnimation();
  };

  tsd::core::logInfo("[planet_demo] Registered planet data animation callback");
}

void PlanetControls::importPlanetData()
{
  auto *core = appCore();
  auto &scene = core->tsd.scene;

  // Only populate once
  if (m_cloudsField)
    return;

  try {
    // Find clouds spatial field
    m_cloudsField = tsd::core::find_item_if(
        scene.objectDB().field, [](const tsd::core::SpatialField *f) {
          return f && f->subtype() == tsd::core::tokens::spatial_field::clouds;
        });

    if (m_cloudsField) {
      // Get number of time steps from metadata
      auto numTimeStepsParam = m_cloudsField->getMetadataValue("numTimeSteps");
      if (numTimeStepsParam.valid()) {
        m_maxTimeSteps = numTimeStepsParam.get<int>();
      } else {
        // Default to 1 time step if metadata not found
        m_maxTimeSteps = 1;
      }
      tsd::core::logInfo(
          "[earth_demo] Found clouds field with %d time steps", m_maxTimeSteps);
    }

    // Find planet spatial field
    m_planetField = tsd::core::find_item_if(
        scene.objectDB().field, [](const tsd::core::SpatialField *f) {
          return f && f->subtype() == tsd::core::tokens::spatial_field::planet;
        });

    if (m_planetField) {
      tsd::core::logInfo("[earth_demo] Found planet field");
    }

    // Find magnetic spatial field
    m_magneticField = tsd::core::find_item_if(
        scene.objectDB().field, [](const tsd::core::SpatialField *f) {
          return f
              && f->subtype() == tsd::core::tokens::spatial_field::magnetic;
        });

    if (m_magneticField) {
      tsd::core::logInfo("[earth_demo] Found magnetic field");
    }

    // Find aurora spatial field
    m_auroraField = tsd::core::find_item_if(
        scene.objectDB().field, [](const tsd::core::SpatialField *f) {
          return f && f->subtype() == tsd::core::tokens::spatial_field::aurora;
        });

    if (m_auroraField) {
      tsd::core::logInfo("[earth_demo] Found aurora field");
    }

    // Check if a directional light already exists
    bool hasDirectionalLight = false;
    tsd::core::foreach_item_const(
        scene.objectDB().light, [&](const tsd::core::Light *light) {
          if (light
              && light->subtype() == tsd::core::tokens::light::directional) {
            hasDirectionalLight = true;
          }
        });

    // Create a directional sun light if none exists
    if (!hasDirectionalLight) {
      auto *layer = scene.defaultLayer();
      auto sunLight = scene.createObject<tsd::core::Light>(
          tsd::core::tokens::light::directional);
      sunLight->setName("sun_light");
      // Direction as azimuth/elevation in degrees (120° azimuth, 45° elevation)
      sunLight->setParameter("direction", tsd::math::float2(120.f, 45.f));
      // Warm sunlight color
      sunLight->setParameter("color", tsd::math::float3(1.0f, 0.95f, 0.8f));
      // Realistic sun irradiance
      sunLight->setParameter("irradiance", 40.0f);
      scene.insertChildObjectNode(layer->root(), sunLight);
      tsd::core::logInfo(
          "[earth_demo] Created directional sun light (az=120°, el=45°, irradiance=40.0)");
    } else {
      tsd::core::logInfo(
          "[earth_demo] Directional light already exists in scene");
    }

    // Setup animation callback
    setupAnimations();

    tsd::core::logInfo("[planet_demo] Planet data initialization complete");
  } catch (const std::exception &e) {
    tsd::core::logError(
        "[planet_demo] Error importing planet data: %s", e.what());
  }
}

} // namespace tsd::demo
