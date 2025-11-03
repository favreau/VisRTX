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

  void setEarthDataFiles(const std::string &cloudsFile,
      const std::string &magneticFile,
      const std::string &planetFile);

  void setAuroraField(tsd::core::SpatialFieldRef auroraField);

  void buildUI() override;

 private:
  void buildUI_incrementAnimation();
  void buildUI_fileSelection();
  void buildUI_animationControls();
  void importEarthData();
  void setTimeStepVolumes();

  // Data //

  std::string m_cloudsFile;
  std::string m_magneticFile;
  std::string m_planetFile;

  tsd::core::Timer m_timer;
  bool m_playing{false};
  float m_targetFps{24.f};

  // Scene objects
  tsd::core::LayerNodeRef m_earthRoot;
  tsd::core::ObjectUsePtr<tsd::core::Volume> m_cloudsVolume;
  tsd::core::ObjectUsePtr<tsd::core::Volume> m_magneticVolume;
  tsd::core::ObjectUsePtr<tsd::core::Volume> m_planetVolume;
  tsd::core::SpatialFieldRef m_auroraField;

  // Time step data (only clouds have time steps)
  std::vector<tsd::core::ObjectUsePtr<tsd::core::SpatialField>>
      m_cloudsTimeSteps;

  int m_currentTimeStep{0};
  int m_maxTimeSteps{0};
};

} // namespace tsd::demo
