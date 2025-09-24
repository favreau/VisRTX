// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "EarthControls.h"
// tsd_core
#include <tsd/core/Logging.hpp>
// tsd_io
#include <tsd/io/importers.hpp>
#include <tsd/io/importers/detail/importer_common.hpp>
// tsd_ui_imgui
#include <tsd/ui/imgui/Application.h>
#include <tsd/ui/imgui/modals/BlockingTaskModal.h>

namespace tsd::demo {

EarthControls::EarthControls(tsd::ui::imgui::Application *app, const char *name)
    : tsd::ui::imgui::Window(app, name)
{}

void EarthControls::setEarthDataFiles(const std::string &cloudsFile,
    const std::string &magneticFile,
    const std::string &planetFile)
{
  m_cloudsFile = cloudsFile;
  m_magneticFile = magneticFile;
  m_planetFile = planetFile;

  auto doLoad = [&]() { importEarthData(); };
  if (!appCore()->windows.taskModal)
    doLoad();
  else {
    appCore()->windows.taskModal->activate(
        doLoad, "Please Wait: Importing Earth Data...");
  }
}

void EarthControls::buildUI()
{
  if (ImGui::IsKeyPressed(ImGuiKey_Space) && m_cloudsVolume)
    m_playing = !m_playing;

  buildUI_incrementAnimation();

  buildUI_fileSelection();
  ImGui::Separator();
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

void EarthControls::buildUI_fileSelection()
{
  constexpr int MAX_LENGTH = 2000;

  // Clouds file selection
  m_cloudsFile.reserve(MAX_LENGTH);
  static std::string cloudsOutPath;
  if (ImGui::Button("Clouds...")) {
    cloudsOutPath.clear();
    m_app->getFilenameFromDialog(cloudsOutPath);
  }
  if (!cloudsOutPath.empty()) {
    m_cloudsFile = cloudsOutPath;
    cloudsOutPath.clear();
  }
  ImGui::SameLine();

  auto text_cb = [](ImGuiInputTextCallbackData *cbd) {
    auto &fname = *(std::string *)cbd->UserData;
    fname.resize(cbd->BufTextLen);
    return 0;
  };

  ImGui::InputText("##cloudsfile",
      m_cloudsFile.data(),
      MAX_LENGTH,
      ImGuiInputTextFlags_CallbackEdit,
      text_cb,
      &m_cloudsFile);

  // Magnetic field file selection
  m_magneticFile.reserve(MAX_LENGTH);
  static std::string magneticOutPath;
  if (ImGui::Button("Magnetic...")) {
    magneticOutPath.clear();
    m_app->getFilenameFromDialog(magneticOutPath);
  }
  if (!magneticOutPath.empty()) {
    m_magneticFile = magneticOutPath;
    magneticOutPath.clear();
  }
  ImGui::SameLine();

  ImGui::InputText("##magneticfile",
      m_magneticFile.data(),
      MAX_LENGTH,
      ImGuiInputTextFlags_CallbackEdit,
      text_cb,
      &m_magneticFile);

  // Planet file selection
  m_planetFile.reserve(MAX_LENGTH);
  static std::string planetOutPath;
  if (ImGui::Button("Planet...")) {
    planetOutPath.clear();
    m_app->getFilenameFromDialog(planetOutPath);
  }
  if (!planetOutPath.empty()) {
    m_planetFile = planetOutPath;
    planetOutPath.clear();
  }
  ImGui::SameLine();

  ImGui::InputText("##planetfile",
      m_planetFile.data(),
      MAX_LENGTH,
      ImGuiInputTextFlags_CallbackEdit,
      text_cb,
      &m_planetFile);

  const bool readyToLoad = !m_cloudsFile.empty() && !m_magneticFile.empty()
      && !m_planetFile.empty() && m_cloudsTimeSteps.empty();

  ImGui::BeginDisabled(!readyToLoad);
  if (ImGui::Button("load earth data"))
    setEarthDataFiles(m_cloudsFile, m_magneticFile, m_planetFile);
  ImGui::EndDisabled();

  ImGui::SameLine();

  ImGui::BeginDisabled(m_cloudsTimeSteps.empty());
  if (ImGui::Button("clear")) {
    m_cloudsTimeSteps.clear();
    if (m_earthRoot)
      (*m_earthRoot)->setEmpty();
    m_currentTimeStep = 0;
    m_maxTimeSteps = 0;
    m_cloudsVolume.reset();
    m_magneticVolume.reset();
    m_planetVolume.reset();
    auto &scene = appCore()->tsd.scene;
    scene.signalLayerChange(scene.defaultLayer());
    scene.cleanupScene();
  }
  ImGui::EndDisabled();
}

void EarthControls::buildUI_animationControls()
{
  ImGui::BeginDisabled(m_playing || m_cloudsTimeSteps.empty());

  if (m_cloudsTimeSteps.empty()) {
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

  ImGui::BeginDisabled(!m_playing || m_cloudsTimeSteps.empty());
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
  auto *layer = core->tsd.scene.defaultLayer();

  // Create earth root transform node
  m_earthRoot = scene.insertChildTransformNode(layer->root());
  (*m_earthRoot)->name() = "earth_transform";

  // Load clouds data
  tsd::core::logInfo(
      "[earth_demo] Loading clouds data from %s", m_cloudsFile.c_str());
  auto cloudsField = tsd::io::import_CLOUDS(scene, m_cloudsFile.c_str());
  if (cloudsField) {
    // Get number of time steps from metadata
    auto numTimeStepsParam = cloudsField->getMetadataValue("numTimeSteps");
    int numTimeSteps = numTimeStepsParam ? numTimeStepsParam.get<int>() : 1;

    tsd::core::logInfo(
        "[earth_demo] Found %d time steps in clouds data", numTimeSteps);

    // For now, just add the first time step - full time series loading would
    // require extending the import functions to accept time indices
    m_cloudsTimeSteps.push_back(cloudsField);

    // Create volume with the field - import_volume creates its own transform,
    // so we'll get a reference to it
    auto cloudsVolume =
        tsd::io::import_volume(scene, m_cloudsFile.c_str(), {}, {});
    m_cloudsVolume = cloudsVolume;
    m_cloudsVolume->setParameter("unitDistance", 8.f);
  }

  // Load magnetic field data (static)
  tsd::core::logInfo("[earth_demo] Loading magnetic field data from %s",
      m_magneticFile.c_str());
  auto magneticVolume =
      tsd::io::import_volume(scene, m_magneticFile.c_str(), {}, {});
  m_magneticVolume = magneticVolume;
  m_magneticVolume->setParameter("unitDistance", 8.f);

  // Load planet data (static)
  tsd::core::logInfo(
      "[earth_demo] Loading planet data from %s", m_planetFile.c_str());
  auto planetVolume =
      tsd::io::import_volume(scene, m_planetFile.c_str(), {}, {});
  m_planetVolume = planetVolume;
  m_planetVolume->setParameter("unitDistance", 8.f);

  // Set max time steps based on clouds data (assuming it drives the animation)
  m_maxTimeSteps = static_cast<int>(m_cloudsTimeSteps.size());
  m_currentTimeStep = 0;

  // Reset camera to fit the loaded data
  core->view.manipulator.setConfig({0.f, 0.f, 0.f}, 5.f, {0.f, 0.f});

  // Notify scene changed
  scene.signalLayerChange(layer);
}

void EarthControls::setTimeStepVolumes()
{
  if (m_currentTimeStep >= m_maxTimeSteps || m_currentTimeStep < 0)
    return;

  if (m_cloudsVolume
      && m_currentTimeStep < static_cast<int>(m_cloudsTimeSteps.size())) {
    auto &scene = appCore()->tsd.scene;

    // Get the spatial field from the volume
    auto valueParam = m_cloudsVolume->parameter("value");
    if (valueParam && valueParam->value().type() == ANARI_OBJECT) {
      auto fieldIndex = valueParam->value().getAsObjectIndex();
      if (auto spatialField =
              scene.getObject<tsd::core::SpatialField>(fieldIndex)) {
        // Update the spatial field with new time step data
        if (tsd::io::update_CLOUDS(
                scene, spatialField, m_cloudsFile.c_str(), m_currentTimeStep)) {
          tsd::core::logInfo(
              "[earth_demo] Updated clouds texture for time step %d",
              m_currentTimeStep);
        }
      }
    }
  }
}

} // namespace tsd::demo
