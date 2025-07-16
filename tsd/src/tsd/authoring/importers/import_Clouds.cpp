// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/authoring/importers.hpp"
#include "tsd/authoring/importers/detail/importer_common.hpp"
#include "tsd/authoring/importers/detail/netcdf_utils.h"
#include "tsd/core/Logging.hpp"
// std
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

using namespace std;
namespace fs = filesystem;

namespace tsd {

struct CloudHeader
{
  // Cloud parameters
  float cloudScale{1.0f};
  float cloudDensity{1.0f};
  float cloudOpacity{0.8f};
  float3 cloudCenter{0.0f, 0.0f, 0.0f};

  // NetCDF file path and variable name
  string netCDFPath;
  string variableName;
  size_t sliceIndex{0}; // For 2D slices from 3D data
  bool useSlice{false}; // Whether to load a 2D slice or full 3D data
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
      if (key == "cloudScale") {
        header.cloudScale = stof(value);
      } else if (key == "cloudDensity") {
        header.cloudDensity = stof(value);
      } else if (key == "cloudOpacity") {
        header.cloudOpacity = stof(value);
      } else if (key == "cloudCenter") {
        sscanf(value.c_str(),
            "{%f, %f, %f}",
            &header.cloudCenter.x,
            &header.cloudCenter.y,
            &header.cloudCenter.z);
      } else if (key == "netCDFPath") {
        header.netCDFPath = value;
      } else if (key == "variableName") {
        header.variableName = value;
      } else if (key == "sliceIndex") {
        header.sliceIndex = stoul(value);
        header.useSlice = true;
      } else if (key == "useSlice") {
        header.useSlice = (value == "true" || value == "1");
      }
    }
  }

  return header;
}

SpatialFieldRef import_Clouds(Context &ctx, const char *filepath)
{
#ifdef TSD_USE_NETCDF
  const auto header = readCloudHeader(filepath);
  const auto basePath = fs::path(filepath).parent_path().string();

  // Create a custom "Cloud" spatial field
  auto field = ctx.createObject<SpatialField>(tokens::spatial_field::planet);
  field->setName(fileOf(filepath).c_str());

  // Log the parsed parameters
  logInfo("[import_Clouds] Cloud parameters loaded:");
  logInfo("  cloudScale: %f", header.cloudScale);
  logInfo("  cloudDensity: %f", header.cloudDensity);
  logInfo("  cloudOpacity: %f", header.cloudOpacity);
  logInfo("  cloudCenter: (%f, %f, %f)",
      header.cloudCenter.x,
      header.cloudCenter.y,
      header.cloudCenter.z);

  if (!header.netCDFPath.empty()) {
    logInfo("  netCDFPath: %s", header.netCDFPath.c_str());
  }
  if (!header.variableName.empty()) {
    logInfo("  variableName: %s", header.variableName.c_str());
  }
  if (header.useSlice) {
    logInfo("  sliceIndex: %zu", header.sliceIndex);
  }

  // Set Cloud-specific parameters
  field->setParameter("cloudScale", header.cloudScale);
  field->setParameter("cloudDensity", header.cloudDensity);
  field->setParameter("cloudOpacity", header.cloudOpacity);
  field->setParameter("cloudCenter", header.cloudCenter);

  field->setParameter("PlanetRadius", 6378000.0f);
  field->setParameter("sphereRadius", 0.99f);
  field->setParameter("elevationScale", 1.f);
  field->setParameter("atmosphereThickness", 0.01f);
  field->setParameter("sphereCenter", float3(0.0f, 0.0f, 0.0f));

  // Load NetCDF data
  if (header.netCDFPath.empty()) {
    logError("[import_Clouds] NetCDF path is required but not provided");
    return {};
  }

  if (header.variableName.empty()) {
    logError("[import_Clouds] Variable name is required but not provided");
    return {};
  }

  string fullNetCDFPath = basePath + "/" + header.netCDFPath;

  // Get variable info for logging
  auto varInfo = getNetCDFVariableInfo(fullNetCDFPath, header.variableName);
  if (varInfo.dimensions.empty()) {
    logError("[import_Clouds] Failed to get variable info for '%s'",
        header.variableName.c_str());
    return {};
  }

  // Load the data
  ArrayRef dataArray;
  if (header.useSlice) {
    // Load 2D slice from 3D data
    dataArray = loadNetCDFSlice(ctx,
        fullNetCDFPath,
        header.variableName,
        "cloudData",
        header.sliceIndex);
  } else {
    // Load full 3D data
    dataArray = loadNetCDFVariable(
        ctx, fullNetCDFPath, header.variableName, "cloudData");
  }

  if (!dataArray) {
    logError("[import_Clouds] Failed to load NetCDF data");
    return {};
  }

  // Set the data as a parameter
  field->setParameterObject("elevationMap", *dataArray);

  logInfo("[import_Clouds] Successfully loaded cloud data from NetCDF file");
  return field;

#else
  logError(
      "[import_Clouds] NetCDF support not enabled. Rebuild with TSD_USE_NETCDF=ON");
  return {};
#endif
}

} // namespace tsd
