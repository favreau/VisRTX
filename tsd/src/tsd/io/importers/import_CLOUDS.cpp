// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

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

  // Log the parsed parameters
  logInfo("[import_Clouds] Cloud parameters loaded:");
  if (!header.netCDFPath.empty()) {
    logInfo("  netCDFPath: %s", header.netCDFPath.c_str());
  }
  if (!header.variableName.empty()) {
    logInfo("  variableName: %s", header.variableName.c_str());
  }

  // Set Cloud-specific parameters
  field->setParameter("planetRadius"_t, header.planetRadius);
  field->setParameter("atmosphereThickness"_t, header.atmosphereThickness);

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

  // Get variable info for logging
  auto varInfo = getNetCDFVariableInfo(fullNetCDFPath, header.variableName);
  if (varInfo.dimensions.empty()) {
    logError("[import_Clouds] Failed to get variable info for '%s'",
        header.variableName.c_str());
    return {};
  }

  // Load the data
  ArrayRef dataArray =
      loadNetCDFVariable(scene, fullNetCDFPath, header.variableName);

  if (!dataArray) {
    logError("[import_Clouds] Failed to load NetCDF data");
    return {};
  }

  // Set the data as a parameter
  field->setParameterObject("cloudData"_t, *dataArray);
  field->setMetadataValue("unitDistance", 256.f);

  logInfo("[import_Clouds] Successfully loaded cloud data from NetCDF file");
  return field;
#else
  logError(
      "[import_Clouds] NetCDF support not enabled. Rebuild with TSD_USE_NETCDF=ON");
  return {};
#endif
}

} // namespace tsd::io
