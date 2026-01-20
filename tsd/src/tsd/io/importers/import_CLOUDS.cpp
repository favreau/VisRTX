// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "import_CLOUDS.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
#include "tsd/io/importers/detail/netcdf_utils.h"
// std
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
// stb_image_write for debug export
#include "stb_image_write.h"

using namespace std;
namespace fs = filesystem;

namespace tsd::io {

struct CloudHeader
{
  float planetRadius{0.905};
  float atmosphereThickness{0.02};
  string netCDFPath;
  string variableName;
  float unitDistance{256.0f};
  string colormap;
  bool debugExportPNG{false}; // Enable debug PNG export
};

CloudHeader readCloudHeader(const string &filename)
{
  CloudHeader header;
  ifstream file(filename);
  string line;

  while (getline(file, line)) {
    // Skip empty lines and comments
    if (line.empty() || line[0] == '#')
      continue;

    // Parse key-value pairs (handle both = and : separators)
    size_t delimPos = line.find("=");
    if (delimPos == string::npos)
      delimPos = line.find(":");

    if (delimPos != string::npos) {
      string key = line.substr(0, delimPos);
      string value = line.substr(delimPos + 1);

      // Trim whitespace
      key.erase(0, key.find_first_not_of(" \t"));
      key.erase(key.find_last_not_of(" \t") + 1);
      value.erase(0, value.find_first_not_of(" \t"));
      value.erase(value.find_last_not_of(" \t") + 1);

      // Parse fields
      if (key == "netCDFPath") {
        header.netCDFPath = value;
      } else if (key == "variableName") {
        header.variableName = value;
      } else if (key == "planetRadius") {
        header.planetRadius = stof(value);
      } else if (key == "atmosphereThickness") {
        header.atmosphereThickness = stof(value);
      } else if (key == "unitDistance") {
        header.unitDistance = stof(value);
      } else if (key == "colormap") {
        header.colormap = value;
      } else if (key == "debugExportPNG") {
        header.debugExportPNG = (value == "true" || value == "1");
      }
    }
  }

  return header;
}

SpatialFieldRef import_CLOUDS(Scene &scene, const char *filepath)
{
#ifdef TSD_USE_NETCDF
  auto header = readCloudHeader(filepath);
  const auto basePath = fs::path(filepath).parent_path().string();

  // Create a custom "Cloud" spatial field
  auto field = scene.createObject<SpatialField>(tokens::spatial_field::clouds);
  field->setName(fileOf(filepath).c_str());

  // Set Cloud-specific parameters
  field->setParameter("planetRadius", header.planetRadius);
  field->setParameter("atmosphereThickness", header.atmosphereThickness);

  // Load NetCDF data
  if (header.netCDFPath.empty()) {
    logError("[import_Clouds] NetCDF path is required but not provided");
    return {};
  }

  if (header.variableName.empty()) {
    logError("[import_Clouds] Variable name is required but not provided");
    return {};
  }

  const string fullNetCDFPath = basePath + "/" + header.netCDFPath;

  // Extract spatial extent from NetCDF coordinate variables
  float minLat = -90.0f, maxLat = 90.0f, minLon = -180.0f, maxLon = 180.0f;
  try {
    netCDF::NcFile file(fullNetCDFPath, netCDF::NcFile::read);

    // Debug: List all variables in the file
    logInfo("[import_CLOUDS] NetCDF variables:");
    auto vars = file.getVars();
    for (const auto &var : vars) {
      std::string varName = var.first;
      auto varObj = var.second;
      logInfo("  - %s (dims: %d)", varName.c_str(), varObj.getDimCount());
    }

    // Try to find latitude coordinate variable
    netCDF::NcVar latVar = file.getVar("lat");
    if (latVar.isNull())
      latVar = file.getVar("latitude");
    if (latVar.isNull())
      latVar = file.getVar("clat");

    // Try to find longitude coordinate variable
    netCDF::NcVar lonVar = file.getVar("lon");
    if (lonVar.isNull())
      lonVar = file.getVar("longitude");
    if (lonVar.isNull())
      lonVar = file.getVar("clon");

    if (!latVar.isNull() && !lonVar.isNull()) {
      size_t latSize = latVar.getDim(0).getSize();
      size_t lonSize = lonVar.getDim(0).getSize();

      logInfo(
          "[import_CLOUDS] Reading lat var '%s' (size=%zu), lon var '%s' (size=%zu)",
          latVar.getName().c_str(),
          latSize,
          lonVar.getName().c_str(),
          lonSize);

      std::vector<double> latData(latSize), lonData(lonSize);
      latVar.getVar(latData.data());
      lonVar.getVar(lonData.data());

      // Debug: Show first few values
      logInfo(
          "[import_CLOUDS] First 5 lat values: %.2f, %.2f, %.2f, %.2f, %.2f",
          latData[0],
          latData[std::min(1ul, latSize - 1)],
          latData[std::min(2ul, latSize - 1)],
          latData[std::min(3ul, latSize - 1)],
          latData[std::min(4ul, latSize - 1)]);
      logInfo(
          "[import_CLOUDS] First 5 lon values: %.2f, %.2f, %.2f, %.2f, %.2f",
          lonData[0],
          lonData[std::min(1ul, lonSize - 1)],
          lonData[std::min(2ul, lonSize - 1)],
          lonData[std::min(3ul, lonSize - 1)],
          lonData[std::min(4ul, lonSize - 1)]);

      // Check if coordinates are in radians (typical for cell-centered data)
      // Use a more robust check: if values are within [-2π, 2π], assume radians
      bool inRadians = (std::abs(latData[0]) <= 2.0 * M_PI
          && std::abs(lonData[0]) <= 2.0 * M_PI);

      logInfo("[import_CLOUDS] inRadians=%d", inRadians);

      if (inRadians) {
        for (auto &val : latData)
          val *= 180.0 / M_PI;
        for (auto &val : lonData)
          val *= 180.0 / M_PI;
      }

      // Find min/max
      auto latMinMax = std::minmax_element(latData.begin(), latData.end());
      auto lonMinMax = std::minmax_element(lonData.begin(), lonData.end());

      minLat = static_cast<float>(*latMinMax.first);
      maxLat = static_cast<float>(*latMinMax.second);
      minLon = static_cast<float>(*lonMinMax.first);
      maxLon = static_cast<float>(*lonMinMax.second);

      logInfo(
          "[import_CLOUDS] Extracted spatial extent: lat[%.2f, %.2f] lon[%.2f, %.2f]",
          minLat,
          maxLat,
          minLon,
          maxLon);
    } else {
      logWarning("[import_CLOUDS] Could not find lat/lon coordinate variables");
    }
  } catch (const netCDF::exceptions::NcException &e) {
    logWarning("[import_CLOUDS] Error reading spatial extent: %s", e.what());
  }

  // Set spatial extent parameters on the field
  field->setParameter("minLat", minLat);
  field->setParameter("maxLat", maxLat);
  field->setParameter("minLon", minLon);
  field->setParameter("maxLon", maxLon);

  // Get variable info to determine number of time steps
  auto varInfo = getNetCDFVariableInfo(fullNetCDFPath, header.variableName);
  if (varInfo.dimensions.empty()) {
    logError("[import_Clouds] Failed to get variable info for '%s'",
        header.variableName.c_str());
    return {};
  }

  // Log dimension information
  logInfo(
      "[import_CLOUDS] Variable '%s' dimensions:", header.variableName.c_str());
  for (size_t i = 0; i < varInfo.dimNames.size(); ++i) {
    logInfo("  [%zu] %s = %zu",
        i,
        varInfo.dimNames[i].c_str(),
        varInfo.dimensions[i]);
  }

  // Find time dimension
  size_t numTimeSteps = 1;
  for (size_t i = 0; i < varInfo.dimNames.size(); ++i) {
    if (varInfo.dimNames[i] == "time") {
      numTimeSteps = varInfo.dimensions[i];
      break;
    }
  }

  // Load the data for first time step
  ArrayRef dataArray =
      loadNetCDFVariable(scene, fullNetCDFPath, header.variableName, 0);

  if (!dataArray) {
    logError("[import_Clouds] Failed to load NetCDF data");
    return {};
  }

  // Set the data as a parameter
  field->setParameterObject("cloudData", *dataArray);

  // Compute data range for automatic colormap scaling
  const float *data = static_cast<const float *>(dataArray->data());
  size_t totalElements =
      dataArray->dim(0) * dataArray->dim(1) * dataArray->dim(2);

  float minVal = std::numeric_limits<float>::max();
  float maxVal = std::numeric_limits<float>::lowest();

  for (size_t i = 0; i < totalElements; ++i) {
    float val = data[i];
    if (std::isfinite(val)) { // Skip NaN and Inf values
      minVal = std::min(minVal, val);
      maxVal = std::max(maxVal, val);
    }
  }

  // Store value range in metadata for volume creation
  if (std::isfinite(minVal) && std::isfinite(maxVal) && minVal < maxVal) {
    field->setMetadataValue("valueRange", tsd::math::float2{minVal, maxVal});
    logInfo("[import_CLOUDS] Data value range: [%.6f, %.6f]", minVal, maxVal);
  } else {
    logWarning(
        "[import_CLOUDS] Could not determine valid value range, using default [0, 1]");
  }

  // Debug: Export first layer as PNG (if enabled in .clouds file)
  if (header.debugExportPNG) {
    size_t width = dataArray->dim(0);
    size_t height = dataArray->dim(1);
    size_t depth = dataArray->dim(2);

    if (width > 0 && height > 0 && depth > 0) {
      logInfo("[import_CLOUDS] Array dimensions: %zux%zux%zu",
          width,
          height,
          depth);

      // Export middle layer
      size_t layerIdx = depth / 2;
      const float *data = static_cast<const float *>(dataArray->data());

      // Convert to 8-bit grayscale
      std::vector<uint8_t> imgData(width * height);
      float minVal = 1e10f, maxVal = -1e10f;

      // Find min/max for normalization
      for (size_t i = 0; i < width * height; ++i) {
        float val = data[layerIdx * width * height + i];
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
      }

      // Normalize and convert
      for (size_t i = 0; i < width * height; ++i) {
        float val = data[layerIdx * width * height + i];
        float normalized =
            (maxVal > minVal) ? (val - minVal) / (maxVal - minVal) : 0.0f;
        imgData[i] = static_cast<uint8_t>(normalized * 255.0f);
      }

      std::string debugPath =
          "/tmp/clouds_debug_layer" + std::to_string(layerIdx) + ".png";
      if (stbi_write_png(
              debugPath.c_str(), width, height, 1, imgData.data(), width)) {
        logInfo(
            "[import_CLOUDS] Debug PNG exported to: %s (range: %.3f - %.3f)",
            debugPath.c_str(),
            minVal,
            maxVal);
      } else {
        logError("[import_CLOUDS] Failed to write debug PNG");
      }
    }
  }

  field->setMetadataValue("filepath", filepath);
  field->setMetadataValue("unitDistance", header.unitDistance);
  field->setMetadataValue("numTimeSteps", static_cast<int>(numTimeSteps));
  return field;
#else
  logError(
      "[import_Clouds] NetCDF support not enabled. Rebuild with TSD_USE_NETCDF=ON");
  return {};
#endif
}

std::string getTransferFunctionName_CLOUDS(const char *filename)
{
  const auto header = readCloudHeader(filename);
  return header.colormap;
}

bool update_CLOUDS(
    Scene &scene, SpatialFieldRef field, const char *filepath, size_t timeIndex)
{
#ifdef TSD_USE_NETCDF
  const auto header = readCloudHeader(filepath);
  const auto basePath = fs::path(filepath).parent_path().string();

  // Load NetCDF data
  if (header.netCDFPath.empty()) {
    logError("[update_Clouds] NetCDF path is required but not provided");
    return false;
  }

  if (header.variableName.empty()) {
    logError("[update_Clouds] Variable name is required but not provided");
    return false;
  }

  const string fullNetCDFPath = basePath + "/" + header.netCDFPath;

  // Get variable info to validate time index
  auto varInfo = getNetCDFVariableInfo(fullNetCDFPath, header.variableName);
  if (varInfo.dimensions.empty()) {
    logError("[update_Clouds] Failed to get variable info for '%s'",
        header.variableName.c_str());
    return false;
  }

  // Find time dimension
  size_t numTimeSteps = 1;
  for (size_t i = 0; i < varInfo.dimNames.size(); ++i) {
    if (varInfo.dimNames[i] == "time") {
      numTimeSteps = varInfo.dimensions[i];
      break;
    }
  }

  // Validate time index
  if (timeIndex >= numTimeSteps) {
    logError("[update_Clouds] Time index %zu out of range (0-%zu)",
        timeIndex,
        numTimeSteps - 1);
    return false;
  }

  // Load the data for the specified time step
  ArrayRef dataArray;
  if (numTimeSteps > 1) {
    dataArray = loadNetCDFVariable(
        scene, fullNetCDFPath, header.variableName, timeIndex);
  } else {
    dataArray =
        loadNetCDFVariable(scene, fullNetCDFPath, header.variableName, 0);
  }

  if (!dataArray) {
    logError(
        "[update_Clouds] Failed to load NetCDF data for time %zu", timeIndex);
    return false;
  }

  // Update the spatial field's cloudData parameter
  field->setParameterObject("cloudData", *dataArray);

  return true;
#else
  logError(
      "[update_Clouds] NetCDF support not enabled. Rebuild with TSD_USE_NETCDF=ON");
  return false;
#endif
}

} // namespace tsd::io
