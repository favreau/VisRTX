// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
#include "tsd/io/importers/import_AURORA.hpp"
#include "tsd/io/importers/import_CLOUDS.hpp"
#include "tsd/io/importers/import_MAGNETIC.hpp"
#include "tsd/io/importers/import_PLANET.hpp"
// std
#include <cstdio>
#include <filesystem>

namespace tsd::io {

using namespace tsd::core;

VolumeRef import_volume(Scene &scene,
    const char *filepath,
    LayerNodeRef location,
    ArrayRef colorArray,
    ArrayRef opacityArray)
{
  SpatialFieldRef field;

  auto file = fileOf(filepath);
  auto ext = extensionOf(filepath);
  if (ext == ".raw")
    field = import_RAW(scene, filepath);
  else if (ext == ".flash" || ext == ".hdf5")
    field = import_FLASH(scene, filepath);
  else if (ext == ".nvdb")
    field = import_NVDB(scene, filepath);
  else if (ext == ".mhd")
    field = import_MHD(scene, filepath);
  else if (ext == ".vtu")
    field = import_VTU(scene, filepath);
  else if (ext == ".vti")
    field = import_VTI(scene, filepath);
  else if (ext == ".clouds")
    field = import_CLOUDS(scene, filepath);
  else if (ext == ".magnetic")
    field = import_MAGNETIC(scene, filepath);
  else if (ext == ".aurora")
    field = import_AURORA(scene, filepath);
  else if (ext == ".planet")
    field = import_PLANET(scene, filepath);
  else {
    logError("[import_volume] no loader for file type '%s'", ext.c_str());
    return {};
  }

  if (!field) {
    logError(
        "[import_volume] unable to load field from file '%s'", file.c_str());
    return {};
  }

  // Try to load colormap from transfer function if not provided
  if (!colorArray) {
    std::string tfName;

    // Get transfer function name based on file type
    if (ext == ".clouds")
      tfName = getTransferFunctionName_CLOUDS(filepath);
    else if (ext == ".planet")
      tfName = getTransferFunctionName_PLANET(filepath);
    else if (ext == ".magnetic")
      tfName = getTransferFunctionName_MAGNETIC(filepath);
    else if (ext == ".aurora")
      tfName = getTransferFunctionName_AURORA(filepath);

    // Try to load the transfer function
    if (!tfName.empty()) {
      const auto basePath =
          std::filesystem::path(filepath).parent_path().string();
      auto tfData = loadTransferFunction(tfName, basePath);
      if (tfData.loaded) {
        colorArray = createColormapArray(scene, tfData);
        logInfo("[import_volume] Loaded colormap '%s' for '%s'",
            tfName.c_str(),
            file.c_str());
      }
    }

    // Fall back to default colormap if transfer function not found
    if (!colorArray) {
      colorArray = scene.createArray(ANARI_FLOAT32_VEC4, 256);
      colorArray->setData(makeDefaultColorMap(colorArray->size()).data());
    }
  }

  float2 valueRange{0.f, 1.f};
  if (field)
    valueRange = field->computeValueRange();

  auto tx = scene.insertChildTransformNode(
      location ? location : scene.defaultLayer()->root());

  // Check if field has scaling factor metadata and apply to transform
  if (field) {
    auto scalingFactorParam = field->getMetadataValue("scale");
    if (scalingFactorParam) {
      auto scaling = scalingFactorParam.get<tsd::math::float3>();
      auto srt = (*tx)->getTransformSRT();
      srt[0] = scaling; // Scale component
      (*tx)->setAsTransform(srt);
      logInfo(
          "[import_volume] Applied scaling factor (%f, %f, %f) to volume transform",
          scaling.x,
          scaling.y,
          scaling.z);
    }
  }

  auto [inst, volume] = scene.insertNewChildObjectNode<Volume>(
      tx, tokens::volume::transferFunction1D);
  volume->setName(fileOf(filepath).c_str());
  volume->setParameterObject("value", *field);
  volume->setParameterObject("color", *colorArray);
  if (opacityArray)
    volume->setParameterObject("opacity", *opacityArray);
  volume->setParameter("valueRange", ANARI_FLOAT32_BOX1, &valueRange);

  // Check if field has unitDistance metadata and set as volume parameter
  if (field) {
    auto unitDistanceParam = field->getMetadataValue("unitDistance");
    if (unitDistanceParam) {
      auto unitDistance = unitDistanceParam.get<float>();
      volume->setParameter("unitDistance", unitDistance);
      logInfo("[import_volume] Set unitDistance parameter to %f", unitDistance);
    }
  }

  return volume;
}

} // namespace tsd::io
