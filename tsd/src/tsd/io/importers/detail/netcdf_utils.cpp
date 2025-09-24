// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "netcdf_utils.h"

#include "tsd/core/scene/Scene.hpp"

#ifdef TSD_USE_NETCDF

#include <algorithm>
#include <cmath>
#include <cstring>
#include <netcdf>

using namespace std;

namespace tsd::io {

NetCDFVariableInfo getNetCDFVariableInfo(
    const string &filepath, const string &variableName)
{
  NetCDFVariableInfo info;

  try {
    netCDF::NcFile file(filepath, netCDF::NcFile::read);
    netCDF::NcVar var = file.getVar(variableName);

    if (var.isNull()) {
      logError("[netcdf_utils] Variable '%s' not found in file '%s'",
          variableName.c_str(),
          filepath.c_str());
      return info;
    }

    info.name = variableName;
    info.dataType = var.getType().getId();

    // Get dimensions
    vector<netCDF::NcDim> dims = var.getDims();
    info.dimensions.resize(dims.size());
    info.dimNames.resize(dims.size());

    for (size_t i = 0; i < dims.size(); ++i) {
      info.dimensions[i] = dims[i].getSize();
      info.dimNames[i] = dims[i].getName();
    }

    // Calculate total elements
    info.totalElements = 1;
    for (size_t dim : info.dimensions) {
      info.totalElements *= dim;
    }

    logInfo("[netcdf_utils] Variable '%s': %zu dimensions, %zu total elements",
        variableName.c_str(),
        info.dimensions.size(),
        info.totalElements);

  } catch (const netCDF::exceptions::NcException &e) {
    logError("[netcdf_utils] NetCDF error reading '%s': %s",
        filepath.c_str(),
        e.what());
  }

  return info;
}

ArrayRef unstructuredDataTo3DTexture(Scene &scene,
    const vector<float> &lonDeg,
    const vector<float> &latDeg,
    const vector<float> &cloudData,
    size_t numCells,
    size_t startLevel,
    size_t endLevel)
{
  // Derive number of levels from data size
  if (cloudData.size() % numCells != 0) {
    logError(
        "[netcdf_utils] Cloud data size %zu is not divisible by numCells %zu",
        cloudData.size(),
        numCells);
    return {};
  }

  size_t numLevels = cloudData.size() / numCells;

  // Validate level range
  if (startLevel >= numLevels || endLevel >= numLevels
      || startLevel > endLevel) {
    logError(
        "[netcdf_utils] Invalid level range [%zu, %zu] for %zu total levels",
        startLevel,
        endLevel,
        numLevels);
    return {};
  }

  size_t levelCount = endLevel - startLevel + 1;

  // Calculate appropriate resolution based on cell density
  float avgCellSpacing = sqrt(360.0f * 180.0f / numCells);
  size_t adaptiveWidth =
      min(1024UL, static_cast<size_t>(360.0f / avgCellSpacing));
  size_t adaptiveHeight =
      min(512UL, static_cast<size_t>(180.0f / avgCellSpacing));

  logInfo(
      "[netcdf_utils] Creating 3D equirectangular texture: %zux%zux%zu (nearest neighbor, levels %zu-%zu)",
      adaptiveWidth,
      adaptiveHeight,
      levelCount,
      startLevel,
      endLevel);

  // Create 3D texture array
  ArrayRef array = scene.createArray(
      ANARI_FLOAT32, adaptiveWidth, adaptiveHeight, levelCount);
  float *texture = reinterpret_cast<float *>(array->map());

  // Initialize texture with fill value
  const float fillValue = 0.f;
  fill(texture,
      texture + (adaptiveWidth * adaptiveHeight * levelCount),
      fillValue);

  logInfo(
      "[netcdf_utils] Processing levels %zu-%zu (%zu levels) with cloudData size: %zu (expected: %zu)",
      startLevel,
      endLevel,
      levelCount,
      cloudData.size(),
      numCells * numLevels);

  // Track statistics
  size_t totalPixels = adaptiveWidth * adaptiveHeight * levelCount;
  size_t pixelsWithData = 0;

  // Track data value range
  float minValue = numeric_limits<float>::max();
  float maxValue = numeric_limits<float>::lowest();
  float minScaledValue = numeric_limits<float>::max();
  float maxScaledValue = numeric_limits<float>::lowest();

  // Simple direct mapping: place each cell's value at its grid position
  for (size_t level = startLevel; level <= endLevel; ++level) {
    for (size_t cellIdx = 0; cellIdx < numCells; ++cellIdx) {
      // Get cell coordinates
      float cellLon = lonDeg[cellIdx];
      float cellLat = latDeg[cellIdx];

      // Normalize longitude to [0, 360) range
      if (cellLon < 0)
        cellLon += 360.0f;

      // Calculate texture coordinates
      float texX = (cellLon / 360.0f) * adaptiveWidth;
      float texY = ((cellLat + 90.0f) / 180.0f) * adaptiveHeight;

      // Clamp to valid range
      size_t gridX =
          min(static_cast<size_t>(max(0.0f, texX)), adaptiveWidth - 1);
      size_t gridY =
          min(static_cast<size_t>(max(0.0f, texY)), adaptiveHeight - 1);

      // Calculate data index for this level and cell
      size_t cloudIdx = level * numCells + cellIdx;

      // Bounds check
      if (cloudIdx < cloudData.size()) {
        // Calculate 3D texture index (Y-flip for correct orientation)
        // Map level to texture Z coordinate (level relative to startLevel)
        size_t texZ = level - startLevel;
        size_t texIdx = texZ * (adaptiveWidth * adaptiveHeight)
            + (adaptiveHeight - 1 - gridY) * adaptiveWidth + gridX;

        // Track raw data range
        float rawValue = cloudData[cloudIdx];
        minValue = min(minValue, rawValue);
        maxValue = max(maxValue, rawValue);

        // Store scaled value
        float scaledValue = rawValue * 1e4f;
        texture[texIdx] = scaledValue;

        // Track scaled data range
        minScaledValue = min(minScaledValue, scaledValue);
        maxScaledValue = max(maxScaledValue, scaledValue);

        pixelsWithData++;
      }
    }
  }

  array->unmap();

  logInfo(
      "[netcdf_utils] 3D texture complete: %zu/%zu pixels have data (%.1f%%)",
      pixelsWithData,
      totalPixels,
      (100.0f * pixelsWithData) / totalPixels);

  if (pixelsWithData > 0) {
    logInfo(
        "[netcdf_utils] 3D data range - Raw: [%.6e, %.6e], Scaled: [%.6e, %.6e]",
        minValue,
        maxValue,
        minScaledValue,
        maxScaledValue);
  }

  return array;
}

ArrayRef loadNetCDFVariable(
    Scene &scene, const string &filepath, const string &variableName)
{
  try {
    netCDF::NcFile file(filepath, netCDF::NcFile::read);
    netCDF::NcVar var = file.getVar(variableName);

    if (var.isNull()) {
      logError("[netcdf_utils] Variable '%s' not found in file '%s'",
          variableName.c_str(),
          filepath.c_str());
      return {};
    }

    // Get dimensions
    vector<netCDF::NcDim> dims = var.getDims();
    logInfo("[netcdf_utils] Variable '%s' has %zu dimensions",
        variableName.c_str(),
        dims.size());

    // Check if this is unstructured data (like ICON model)
    bool isUnstructured = false;
    size_t numCells = 0;
    size_t numLevels = 0;
    size_t numTimes = 1;

    for (const auto &dim : dims) {
      string dimName = dim.getName();
      size_t dimSize = dim.getSize();
      logInfo("[netcdf_utils] Dimension '%s': %zu", dimName.c_str(), dimSize);

      if (dimName == "cell") {
        isUnstructured = true;
        numCells = dimSize;
      } else if (dimName == "height" || dimName == "lev"
          || dimName == "level") {
        numLevels = dimSize;
      } else if (dimName == "time") {
        numTimes = dimSize;
      }
    }

    if (isUnstructured) {
      logInfo(
          "[netcdf_utils] Detected unstructured grid with %zu cells and %zu levels",
          numCells,
          numLevels);
    } else {
      // Fallback to structured grid handling
      if (dims.size() != 3) {
        logError(
            "[netcdf_utils] Variable '%s' has %zu dimensions, expected 3 for structured grid",
            variableName.c_str(),
            dims.size());
        return {};
      }

      size_t width = dims[0].getSize();
      size_t height = dims[1].getSize();
      size_t depth = dims[2].getSize();

      logInfo("[netcdf_utils] Loading structured 3D variable '%s': %zux%zux%zu",
          variableName.c_str(),
          width,
          height,
          depth);
    }

    // Create array based on data type
    ArrayRef array;
    nc_type dataType = var.getType().getId();

    if (dataType == NC_FLOAT) {
      if (isUnstructured) {
        // Handle unstructured grid (ICON model format)
        logInfo("[netcdf_utils] Processing unstructured grid data");

        // Read longitude and latitude coordinates (in radians)
        netCDF::NcVar clonVar = file.getVar("clon");
        netCDF::NcVar clatVar = file.getVar("clat");

        if (clonVar.isNull() || clatVar.isNull()) {
          logError(
              "[netcdf_utils] Could not find clon/clat coordinate variables");
          return {};
        }

        vector<double> lonRad(numCells), latRad(numCells);
        clonVar.getVar(lonRad.data());
        clatVar.getVar(latRad.data());

        // Read height coordinate data
        netCDF::NcVar heightVar = file.getVar("height");
        vector<double> heightData(numLevels);
        if (!heightVar.isNull()) {
          heightVar.getVar(heightData.data());
        } else {
          // Fallback: use level indices as height values
          for (size_t i = 0; i < numLevels; ++i) {
            heightData[i] = static_cast<double>(i);
          }
          logInfo(
              "[netcdf_utils] No height coordinate found, using level indices");
        }

        // Read cloud data for all height levels
        vector<float> cloudData(numCells * numLevels);
        vector<size_t> start = {0, 0, 0}; // time=0, height=0, cell=0
        vector<size_t> count = {
            1, numLevels, numCells}; // 1 time, all heights, all cells
        var.getVar(start, count, cloudData.data());

        // Convert coordinates from radians to degrees
        vector<float> lonDeg(numCells), latDeg(numCells);
        for (size_t i = 0; i < numCells; ++i) {
          lonDeg[i] = static_cast<float>(lonRad[i] * 180.0 / M_PI);
          latDeg[i] = static_cast<float>(latRad[i] * 180.0 / M_PI);

          // Normalize longitude to [0, 360) range
          if (lonDeg[i] < 0)
            lonDeg[i] += 360.0f;
        }

        // Use appropriate processing function based on interpolation setting
        // Process a subset of levels for 3D texture (levels 10-30 as example)
        // Derive numLevels from cloudData for range calculation
        size_t inferredLevels = cloudData.size() / numCells;
        size_t startLevel = min(0UL, inferredLevels - 1);
        size_t endLevel = min(inferredLevels - 1, inferredLevels - 1);
        array = unstructuredDataTo3DTexture(
            scene, lonDeg, latDeg, cloudData, numCells, startLevel, endLevel);

        logInfo(
            "[netcdf_utils] Successfully created 3D equirectangular texture from %zu unstructured cells across %zu height levels",
            numCells,
            numLevels);
      } else {
        // Handle structured grid (original code path)
        size_t width = dims[0].getSize();
        size_t height = dims[1].getSize();
        size_t depth = dims[2].getSize();

        // Read height data from first slice (depth=0) of 3D data
        vector<float> heightData(width * height);
        vector<size_t> start = {0, 0, 0};
        vector<size_t> count = {width, height, 1};
        var.getVar(start, count, heightData.data());

        // Extract longitude and latitude coordinate information
        vector<float> lonData(width);
        vector<float> latData(height);
        bool hasCoordinates = false;

        try {
          // Try common coordinate variable names
          vector<string> lonNames = {"longitude", "lon", "x"};
          vector<string> latNames = {"latitude", "lat", "y"};

          netCDF::NcVar lonVar, latVar;

          // Find longitude variable
          for (const auto &lonName : lonNames) {
            lonVar = file.getVar(lonName);
            if (!lonVar.isNull()) {
              logInfo("[netcdf_utils] Found longitude variable: %s",
                  lonName.c_str());
              break;
            }
          }

          // Find latitude variable
          for (const auto &latName : latNames) {
            latVar = file.getVar(latName);
            if (!latVar.isNull()) {
              logInfo("[netcdf_utils] Found latitude variable: %s",
                  latName.c_str());
              break;
            }
          }

          if (!lonVar.isNull() && !latVar.isNull()) {
            // Read coordinate data
            lonVar.getVar(lonData.data());
            latVar.getVar(latData.data());
            hasCoordinates = true;

            logInfo(
                "[netcdf_utils] Successfully loaded longitude (%zu values) and latitude (%zu values)",
                width,
                height);
          }
        } catch (const netCDF::exceptions::NcException &e) {
          logInfo("[netcdf_utils] Could not load coordinate variables: %s",
              e.what());
        }

        // Create 2D buffer with longitude, latitude, and height
        array = scene.createArray(ANARI_FLOAT32_VEC3, width, height);
        float3 *buffer = reinterpret_cast<float3 *>(array->map());

        for (size_t j = 0; j < height; ++j) {
          for (size_t i = 0; i < width; ++i) {
            size_t idx = j * width + i;

            if (hasCoordinates) {
              // Use actual longitude/latitude coordinates
              buffer[idx] = {lonData[i], latData[j], heightData[idx]};
            } else {
              // Fallback to grid indices if no coordinates found
              buffer[idx] = {static_cast<float>(i),
                  static_cast<float>(j),
                  heightData[idx]};
            }
          }
        }

        array->unmap();

        if (!hasCoordinates) {
          logInfo(
              "[netcdf_utils] Using grid indices as coordinates (no coordinate variables found)");
        }
        logInfo(
            "[netcdf_utils] Created 2D buffer (%zux%zu) with longitude, latitude, and height data",
            width,
            height);
      }
    } else {
      logError("[netcdf_utils] Unsupported data type %d for variable '%s'",
          dataType,
          variableName.c_str());
      return {};
    }

    logInfo("[netcdf_utils] Successfully loaded as %s", variableName.c_str());
    return array;
  } catch (const netCDF::exceptions::NcException &e) {
    logError("[netcdf_utils] NetCDF error loading '%s': %s",
        filepath.c_str(),
        e.what());
    return {};
  }
}
} // namespace tsd::io
#endif // TSD_USE_NETCDF
