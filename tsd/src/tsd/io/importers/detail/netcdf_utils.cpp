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

    // Display all available variables in the file
    auto vars = file.getVars();

    for (const auto &varPair : vars) {
      const string &varName = varPair.first;
      const netCDF::NcVar &variable = varPair.second;

      // Get variable dimensions
      vector<netCDF::NcDim> varDims = variable.getDims();
      string dimStr = "";
      for (size_t i = 0; i < varDims.size(); ++i) {
        if (i > 0)
          dimStr += ", ";
        dimStr += varDims[i].getName() + "("
            + std::to_string(varDims[i].getSize()) + ")";
      }

      // Get data type name
      string typeName;
      nc_type varType = variable.getType().getId();
      switch (varType) {
      case NC_FLOAT:
        typeName = "float";
        break;
      case NC_DOUBLE:
        typeName = "double";
        break;
      case NC_INT:
        typeName = "int";
        break;
      case NC_SHORT:
        typeName = "short";
        break;
      case NC_CHAR:
        typeName = "char";
        break;
      case NC_BYTE:
        typeName = "byte";
        break;
      default:
        typeName = "unknown";
        break;
      }
    }

    // Now get the specific variable requested
    netCDF::NcVar var = file.getVar(variableName);

    if (var.isNull()) {
      logError("[netcdf_utils] Variable '%s' not found in file '%s'",
          variableName.c_str(),
          filepath.c_str());
      logError("[netcdf_utils] Available variables listed above.");
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

  // Create 3D texture array
  ArrayRef array = scene.createArray(
      ANARI_FLOAT32, adaptiveWidth, adaptiveHeight, levelCount);
  float *texture = reinterpret_cast<float *>(array->map());

  // Initialize texture with fill value
  const float fillValue = 0.f;
  fill(texture,
      texture + (adaptiveWidth * adaptiveHeight * levelCount),
      fillValue);

  // Track statistics
  size_t totalPixels = adaptiveWidth * adaptiveHeight * levelCount;
  size_t pixelsWithData = 0;

  // First pass: find data value range
  float minValue = numeric_limits<float>::max();
  float maxValue = numeric_limits<float>::lowest();

  for (size_t level = startLevel; level <= endLevel; ++level) {
    for (size_t cellIdx = 0; cellIdx < numCells; ++cellIdx) {
      size_t cloudIdx = level * numCells + cellIdx;
      if (cloudIdx < cloudData.size()) {
        float rawValue = cloudData[cloudIdx];
        minValue = min(minValue, rawValue);
        maxValue = max(maxValue, rawValue);
      }
    }
  }

  // Calculate normalization parameters
  float valueRange = maxValue - minValue;
  bool hasValidRange = (valueRange > 1e-10f);

  // Second pass: normalize and place values
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

        float rawValue = cloudData[cloudIdx];

        // Normalize value to [0, 1] range based on actual data range
        float normalizedValue;
        if (hasValidRange) {
          normalizedValue = (rawValue - minValue) / valueRange;
        } else {
          // If all values are the same, use 0.5 as normalized value
          normalizedValue = 0.5f;
        }

        texture[texIdx] = normalizedValue;
        pixelsWithData++;
      }
    }
  }

  array->unmap();

  return array;
}

ArrayRef loadNetCDFVariable(Scene &scene,
    const std::string &filepath,
    const std::string &variableName,
    size_t timeIndex)
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
    //     dims.size());

    // Check if this is unstructured data (like ICON model)
    bool isUnstructured = false;
    size_t numCells = 0;
    size_t numLevels = 0;
    size_t numTimes = 1;

    for (const auto &dim : dims) {
      string dimName = dim.getName();
      size_t dimSize = dim.getSize();

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

    // Validate timeIndex
    if (timeIndex >= numTimes) {
      logError("[netcdf_utils] Time index %zu out of range (0-%zu)",
          timeIndex,
          numTimes - 1);
      return {};
    }

    // Create array based on data type
    ArrayRef array;
    nc_type dataType = var.getType().getId();

    if (dataType == NC_FLOAT) {
      if (isUnstructured) {
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
        }

        // Read cloud data for all height levels at specific time
        vector<float> cloudData(numCells * numLevels);
        vector<size_t> start = {
            timeIndex, 0, 0}; // specific time, height=0, cell=0
        vector<size_t> count = {
            1, numLevels, numCells}; // 1 time, all heights, all cells
        var.getVar(start, count, cloudData.data());

        // Convert coordinates from radians to degrees
        vector<float> lonDeg(numCells), latDeg(numCells);
        for (size_t i = 0; i < numCells; ++i) {
          lonDeg[i] = static_cast<float>(lonRad[i] * 180.0 / M_PI);
          latDeg[i] = static_cast<float>(latRad[i] * 180.0 / M_PI);
        }

        // Convert to 3D texture
        array = unstructuredDataTo3DTexture(
            scene, lonDeg, latDeg, cloudData, numCells, 0, numLevels - 1);
        return array;
      } else {
        // Determine grid dimensions (assume typical structured grid layout)
        size_t nx = 0, ny = 0, nz = 0;
        bool hasTime = false;

        for (const auto &dim : dims) {
          string dimName = dim.getName();
          size_t dimSize = dim.getSize();

          if (dimName == "time") {
            hasTime = true;
          } else if (dimName == "lon" || dimName == "longitude"
              || dimName == "x") {
            nx = dimSize;
          } else if (dimName == "lat" || dimName == "latitude"
              || dimName == "y") {
            ny = dimSize;
          } else if (dimName == "lev" || dimName == "level"
              || dimName == "height" || dimName == "z") {
            nz = dimSize;
          }
        }

        // Calculate total size more carefully
        vector<size_t> start, count;
        size_t totalDataSize = 1;

        // Build start/count arrays to match NetCDF variable dimension order
        // We need to respect the actual dimension order in the file
        vector<size_t> spatialDims;
        for (const auto &dim : dims) {
          string dimName = dim.getName();
          size_t dimSize = dim.getSize();

          if (dimName == "time") {
            start.push_back(timeIndex);
            count.push_back(1);
            // Don't include time dimension in data size for single time step
          } else {
            start.push_back(0);
            count.push_back(dimSize);
            totalDataSize *= dimSize;
            spatialDims.push_back(dimSize);
          }
        }

        // Validate arrays match variable dimensions
        if (start.size() != dims.size() || count.size() != dims.size()) {
          logError(
              "[netcdf_utils] Array size mismatch: var has %zu dims, start=%zu, count=%zu",
              dims.size(),
              start.size(),
              count.size());
          return {};
        }

        // Validate data size is reasonable
        if (totalDataSize == 0 || totalDataSize > 1e9) {
          logError("[netcdf_utils] Invalid data size: %zu", totalDataSize);
          return {};
        }

        // Read data for specific time step
        vector<float> data(totalDataSize);

        try {
          if (hasTime && numTimes > 1) {
            // Multi-dimensional array with time - use hyperslab
            var.getVar(start, count, data.data());
          } else {
            // No time dimension or single time step
            var.getVar(data.data());
          }
        } catch (const netCDF::exceptions::NcException &e) {
          logError("[netcdf_utils] NetCDF read error: %s", e.what());
          return {};
        }

        // Create ANARI array using NetCDF file dimension order to match data
        // layout
        if (spatialDims.size() == 3) {
          // 3D array: use file order
          array = scene.createArray(
              ANARI_FLOAT32, spatialDims[2], spatialDims[1], spatialDims[0]);
        } else if (spatialDims.size() == 2) {
          // 2D array: reverse for (lon,lat) order
          array =
              scene.createArray(ANARI_FLOAT32, spatialDims[1], spatialDims[0]);
        } else if (spatialDims.size() == 1) {
          // 1D array
          array = scene.createArray(ANARI_FLOAT32, spatialDims[0]);
        } else {
          logError(
              "[netcdf_utils] No valid spatial dimensions found: %zu spatial dims",
              spatialDims.size());
          return {};
        }

        array->setData(data.data());
      }
    } else {
      logError("[netcdf_utils] Unsupported data type %d for variable '%s'",
          dataType,
          variableName.c_str());
      return {};
    }

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
