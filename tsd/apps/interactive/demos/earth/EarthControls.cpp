// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "EarthControls.h"
// tsd_core
#include <tsd/core/Logging.hpp>
// tsd_io
#include <tsd/io/importers.hpp>
#include <tsd/io/importers/detail/importer_common.hpp>
#include <tsd/io/importers/import_CLOUDS.hpp>
#include <tsd/io/importers/import_MAGNETIC.hpp>
#include <tsd/io/importers/import_PLANET.hpp>
// tsd_ui_imgui
#include <tsd/ui/imgui/Application.h>
#include <tsd/ui/imgui/modals/BlockingTaskModal.h>
// std
#include <filesystem>
// cuda
#include <cuda_runtime.h>

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

  try {
    // Load clouds data
    tsd::core::logInfo(
        "[earth_demo] Loading clouds data from %s", m_cloudsFile.c_str());
    int numTimeSteps = 1;
    auto cloudsField = tsd::io::import_CLOUDS(scene, m_cloudsFile.c_str());
    if (cloudsField) {
      // Get number of time steps from metadata
      auto numTimeStepsParam = cloudsField->getMetadataValue("numTimeSteps");
      numTimeSteps = numTimeStepsParam ? numTimeStepsParam.get<int>() : 1;

      tsd::core::logInfo(
          "[earth_demo] Found %d time steps in clouds data", numTimeSteps);

      // Store the spatial field for time series updates
      m_cloudsTimeSteps.push_back(cloudsField);

      // Load colormap and create volume with the field
      auto cloudsTF =
          tsd::io::getTransferFunctionName_CLOUDS(m_cloudsFile.c_str());
      tsd::core::ArrayRef cloudsColorArray;
      if (!cloudsTF.empty()) {
        const auto basePath =
            std::filesystem::path(m_cloudsFile).parent_path().string();
        auto tfData = tsd::io::loadTransferFunction(cloudsTF, basePath);
        if (tfData.loaded) {
          cloudsColorArray = tsd::io::createColormapArray(scene, tfData);
          tsd::core::logInfo(
              "[earth_demo] Loaded colormap '%s' for clouds", cloudsTF.c_str());
        } else {
          tsd::core::logWarning(
              "[earth_demo] Failed to load colormap '%s' for clouds",
              cloudsTF.c_str());
        }
      }
      auto cloudsVolume = tsd::io::import_volume(
          scene, m_cloudsFile.c_str(), cloudsColorArray, {});
      m_cloudsVolume = cloudsVolume;
      // Get unitDistance from the spatial field metadata
      auto unitDistanceParam = cloudsField->getMetadataValue("unitDistance");
      if (unitDistanceParam) {
        auto unitDistance = unitDistanceParam.get<float>();
        m_cloudsVolume->setParameter("unitDistance", unitDistance);
      }
    }

    // Load magnetic field data (static)
    tsd::core::logInfo("[earth_demo] Loading magnetic field data from %s",
        m_magneticFile.c_str());
    auto magneticTF =
        tsd::io::getTransferFunctionName_MAGNETIC(m_magneticFile.c_str());
    tsd::core::ArrayRef magneticColorArray;
    if (!magneticTF.empty()) {
      const auto basePath =
          std::filesystem::path(m_magneticFile).parent_path().string();
      auto tfData = tsd::io::loadTransferFunction(magneticTF, basePath);
      if (tfData.loaded) {
        magneticColorArray = tsd::io::createColormapArray(scene, tfData);
        tsd::core::logInfo(
            "[earth_demo] Loaded colormap '%s' for magnetic field",
            magneticTF.c_str());
      } else {
        tsd::core::logWarning(
            "[earth_demo] Failed to load colormap '%s' for magnetic field",
            magneticTF.c_str());
      }
    }
    auto magneticVolume = tsd::io::import_volume(
        scene, m_magneticFile.c_str(), magneticColorArray, {});
    m_magneticVolume = magneticVolume;

    // Get magnetic field for unitDistance
    auto magneticField =
        tsd::io::import_MAGNETIC(scene, m_magneticFile.c_str());
    if (magneticField) {
      auto unitDistanceParam = magneticField->getMetadataValue("unitDistance");
      if (unitDistanceParam) {
        auto unitDistance = unitDistanceParam.get<float>();
        m_magneticVolume->setParameter("unitDistance", unitDistance);
      }
    }

    // Load planet data (static)
    tsd::core::logInfo(
        "[earth_demo] Loading planet data from %s", m_planetFile.c_str());
    auto planetTF =
        tsd::io::getTransferFunctionName_PLANET(m_planetFile.c_str());
    tsd::core::ArrayRef planetColorArray;
    if (!planetTF.empty()) {
      const auto basePath =
          std::filesystem::path(m_planetFile).parent_path().string();
      auto tfData = tsd::io::loadTransferFunction(planetTF, basePath);
      if (tfData.loaded) {
        planetColorArray = tsd::io::createColormapArray(scene, tfData);
        tsd::core::logInfo(
            "[earth_demo] Loaded colormap '%s' for planet", planetTF.c_str());
      } else {
        tsd::core::logWarning(
            "[earth_demo] Failed to load colormap '%s' for planet",
            planetTF.c_str());
      }
    }
    auto planetVolume = tsd::io::import_volume(
        scene, m_planetFile.c_str(), planetColorArray, {});
    m_planetVolume = planetVolume;

    // Get planet field for unitDistance
    auto planetField = tsd::io::import_PLANET(scene, m_planetFile.c_str());
    if (planetField) {
      auto unitDistanceParam = planetField->getMetadataValue("unitDistance");
      if (unitDistanceParam) {
        auto unitDistance = unitDistanceParam.get<float>();
        m_planetVolume->setParameter("unitDistance", unitDistance);
      }
    }

    // Set max time steps from clouds data
    m_maxTimeSteps = numTimeSteps;
    m_currentTimeStep = 0;

    // Reset camera to fit the loaded data
    core->view.manipulator.setConfig({0.f, 0.f, 0.f}, 5.f, {0.f, 0.f});

    // Add a directional light for Earth visualization
    auto sunLight = scene.createObject<tsd::core::Light>(
        tsd::core::tokens::light::directional);
    sunLight->setName("sun_light");
    sunLight->setParameter("direction", tsd::math::float2(120.f, 175.f));
    sunLight->setParameter(
        "color", tsd::math::float3(1.f, 0.95f, 0.8f)); // Warm sunlight color
    sunLight->setParameter("irradiance", 40.0f);
    scene.insertChildObjectNode(layer->root(), sunLight);
    tsd::core::logInfo(
        "[earth_demo] Added sun light: direction(120°, 175°), color(1.0, 0.95, 0.8), irradiance=40.0");

    // Notify scene changed
    scene.signalLayerChange(layer);
  } catch (const std::exception &e) {
    tsd::core::logError(("Error: " + std::string(e.what())).c_str());
  }
}

void EarthControls::setTimeStepVolumes()
{
  if (m_currentTimeStep >= m_maxTimeSteps || m_currentTimeStep < 0)
    return;

  if (m_cloudsVolume) {
    auto &scene = appCore()->tsd.scene;

    // Get the spatial field from the volume's "value" parameter
    auto valueParam = m_cloudsVolume->parameter("value");
    if (valueParam) {
      auto fieldIndex = valueParam->value().getAsObjectIndex();
      auto spatialField = scene.getObject<tsd::core::SpatialField>(fieldIndex);
      if (tsd::io::update_CLOUDS(
              scene, spatialField, m_cloudsFile.c_str(), m_currentTimeStep)) {
        // tsd::core::logInfo(
        //     "[earth_demo] Updated clouds texture for time step %d",
        //     m_currentTimeStep);
      }
    }
  }
}

} // namespace tsd::demo
