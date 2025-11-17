// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "EarthControls.h"
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

EarthControls::EarthControls(tsd::ui::imgui::Application *app, const char *name)
    : tsd::ui::imgui::Window(app, name)
{}

void EarthControls::buildUI()
{
  if (ImGui::IsKeyPressed(ImGuiKey_Space))
    m_playing = !m_playing;

  buildUI_incrementAnimation();

  buildUI_animationControls();
}

void EarthControls::buildUI_incrementAnimation()
{
  if (!m_playing || m_maxTimeSteps == 0)
    return;

  m_timer.end();
  if (auto fps = m_timer.perSecond(); m_targetFps >= fps) {
    m_currentTimeStep++;
    m_currentTimeStep = int(m_currentTimeStep % m_maxTimeSteps);
    setTimeStepVolumes();
  }
}

void EarthControls::buildUI_animationControls()
{
  const bool hasEarthData = m_cloudsField && m_maxTimeSteps > 0;

  ImGui::BeginDisabled(m_playing || !hasEarthData);

  if (!hasEarthData) {
    ImGui::Text("no earth data loaded");
    ImGui::EndDisabled();
    return;
  }

  if (ImGui::SliderInt("time step", &m_currentTimeStep, 0, m_maxTimeSteps - 1))
    setTimeStepVolumes();

  ImGui::SameLine();
  ImGui::Text("/ %d", m_maxTimeSteps - 1);

  if (ImGui::Button("play")) {
    m_timer.start();
    m_playing = true;
  }
  ImGui::EndDisabled();

  ImGui::BeginDisabled(!m_playing || !hasEarthData);
  ImGui::SameLine();
  if (ImGui::Button("stop"))
    m_playing = false;
  ImGui::EndDisabled();

  ImGui::SameLine();
  ImGui::DragFloat("fps", &m_targetFps, 0.01f, 1.f, 120.f);
}

void EarthControls::importEarthData()
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

    // Initialize to first timestep
    m_currentTimeStep = 0;

    tsd::core::logInfo("[earth_demo] Earth data initialization complete");
  } catch (const std::exception &e) {
    tsd::core::logError(
        "[earth_demo] Error importing earth data: %s", e.what());
  }
}

void EarthControls::setTimeStepVolumes()
{
  if (m_currentTimeStep >= m_maxTimeSteps || m_currentTimeStep < 0)
    return;

  auto &scene = appCore()->tsd.scene;

  // Update clouds data using update_CLOUDS function
  if (m_cloudsField) {
    // Get file path from metadata
    auto filepathParam = m_cloudsField->getMetadataValue("filepath");
    if (filepathParam.valid()) {
      std::string cloudsFile = filepathParam.getString();
      if (tsd::io::update_CLOUDS(
              scene, m_cloudsField, cloudsFile.c_str(), m_currentTimeStep)) {
        tsd::core::logInfo(
            "[earth_demo] Updated clouds to time step %d", m_currentTimeStep);
        scene.signalLayerChange(scene.defaultLayer());
      } else {
        tsd::core::logWarning(
            "[earth_demo] Failed to update clouds to time step %d",
            m_currentTimeStep);
      }
    }
  }

  // Update aurora time parameter with current timestep
  if (m_auroraField) {
    // Normalize time step to 0.0-1.0 range
    float normalizedTime = m_maxTimeSteps > 1
        ? static_cast<float>(m_currentTimeStep)
            / static_cast<float>(m_maxTimeSteps - 1)
        : 0.0f;

    m_auroraField->setParameter("time", normalizedTime * 100.f);
  }
}

} // namespace tsd::demo
