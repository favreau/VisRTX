// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Logging.hpp"
#include "tsd/core/scene/objects/Array.hpp"
#include "tsd/core/scene/objects/SpatialField.hpp"

#ifdef TSD_USE_NETCDF
#include <netcdf>
#endif

namespace tsd::io {

#ifdef TSD_USE_NETCDF

using namespace tsd::core;

struct NetCDFVariableInfo
{
  std::string name;
  std::vector<size_t> dimensions;
  std::vector<std::string> dimNames;
  nc_type dataType;
  size_t totalElements;
};

// Get information about a variable in a netCDF file
NetCDFVariableInfo getNetCDFVariableInfo(
    const std::string &filepath, const std::string &variableName);

// Load a 3D variable from netCDF file into an ANARI array
ArrayRef loadNetCDFVariable(
    Scene &scene, const std::string &filepath, const std::string &variableName);

// Load a 3D variable from netCDF file for a specific timestamp
ArrayRef loadNetCDFVariable(Scene &scene,
    const std::string &filepath,
    const std::string &variableName,
    size_t timeIndex);

// Convert unstructured data to 3D equirectangular texture using nearest
// neighbor (for specified level range)
ArrayRef unstructuredDataTo3DTexture(Scene &scene,
    const std::vector<float> &lonDeg,
    const std::vector<float> &latDeg,
    const std::vector<float> &cloudData,
    size_t numCells,
    size_t startLevel,
    size_t endLevel);

// Convert unstructured data to 2D equirectangular texture using nearest
// neighbor (single level)
ArrayRef nearestNeighborUnstructuredDataToTexture2D(Scene &scene,
    const std::vector<float> &lonDeg,
    const std::vector<float> &latDeg,
    const std::vector<float> &cloudData,
    size_t numCells,
    size_t numLevels,
    size_t level);

#endif // TSD_USE_NETCDF
} // namespace tsd::io
