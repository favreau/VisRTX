// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_app
#include <tsd/app/Core.h>
// tsd_ui_imgui
#include <tsd/ui/imgui/windows/Window.h>
// tsd_core
#include <tsd/core/Timer.hpp>
// std
#include <string>
#include <vector>

namespace tsd::demo {

struct EarthControls : public tsd::ui::imgui::Window
{
  EarthControls(
      tsd::ui::imgui::Application *app, const char *name = "Earth Controls");

  void buildUI() override;

  // Call this once after scene data is loaded
  void importEarthData();

 private:
  void buildUI_incrementAnimation();
  void buildUI_animationControls();
  void setTimeStepVolumes();

  tsd::core::Timer m_timer;
  bool m_playing{false};
  float m_targetFps{24.f};

  // Spatial field references
  tsd::core::SpatialFieldRef m_cloudsField;
  tsd::core::SpatialFieldRef m_magneticField;
  tsd::core::SpatialFieldRef m_planetField;
  tsd::core::SpatialFieldRef m_auroraField;

  int m_currentTimeStep{0};
  int m_maxTimeSteps{0};
};

} // namespace tsd::demo
