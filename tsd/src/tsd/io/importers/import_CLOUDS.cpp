// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "import_CLOUDS.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
#include "tsd/io/importers/detail/netcdf_utils.h"
// std
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

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
      }
    }
  }

  return header;
}

SpatialFieldRef import_CLOUDS(Scene &scene, const char *filepath)
{
#ifdef TSD_USE_NETCDF
  const auto header = readCloudHeader(filepath);
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

  // Get variable info to determine number of time steps
  auto varInfo = getNetCDFVariableInfo(fullNetCDFPath, header.variableName);
  if (varInfo.dimensions.empty()) {
    logError("[import_Clouds] Failed to get variable info for '%s'",
        header.variableName.c_str());
    return {};
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
