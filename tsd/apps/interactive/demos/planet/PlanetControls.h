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

struct PlanetControls : public tsd::ui::imgui::Window
{
  PlanetControls(
      tsd::ui::imgui::Application *app, const char *name = "Planet Controls");

  void buildUI() override;

  // Call this once after scene data is loaded
  void importPlanetData();

  // Setup animation for clouds and aurora
  void setupAnimations();

 private:
  void updateTimeStepFromAnimation();

  // Spatial field references
  tsd::core::SpatialFieldRef m_cloudsField;
  tsd::core::SpatialFieldRef m_magneticField;
  tsd::core::SpatialFieldRef m_planetField;
  tsd::core::SpatialFieldRef m_auroraField;

  int m_maxTimeSteps{0};
  float m_lastAnimationTime{-1.f};
};

} // namespace tsd::demo
