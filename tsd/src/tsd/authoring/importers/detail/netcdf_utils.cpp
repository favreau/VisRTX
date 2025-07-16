// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "netcdf_utils.h"

#ifdef TSD_USE_NETCDF

#include <algorithm>
#include <cmath>
#include <cstring>
#include <netcdf>

namespace tsd {

NetCDFVariableInfo getNetCDFVariableInfo(
    const std::string &filepath, const std::string &variableName)
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
    std::vector<netCDF::NcDim> dims = var.getDims();
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

ArrayRef loadNetCDFVariable(Context &ctx,
    const std::string &filepath,
    const std::string &variableName,
    const std::string &paramName)
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
    std::vector<netCDF::NcDim> dims = var.getDims();
    logInfo("[netcdf_utils] Variable '%s' has %zu dimensions",
        variableName.c_str(),
        dims.size());

    // Check if this is unstructured data (like ICON model)
    bool isUnstructured = false;
    size_t numCells = 0;
    size_t numLevels = 0;
    size_t numTimes = 1;

    for (const auto &dim : dims) {
      std::string dimName = dim.getName();
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

        std::vector<double> lonRad(numCells), latRad(numCells);
        clonVar.getVar(lonRad.data());
        clatVar.getVar(latRad.data());

        // Read cloud data for first height level (level 0)
        std::vector<float> cloudData(numCells);
        std::vector<size_t> start = {0, 25, 0}; // time=0, height=0, cell=0
        std::vector<size_t> count = {
            1, 1, numCells}; // 1 time, 1 height, all cells
        var.getVar(start, count, cloudData.data());

        // Convert coordinates from radians to degrees
        std::vector<float> lonDeg(numCells), latDeg(numCells);
        for (size_t i = 0; i < numCells; ++i) {
          lonDeg[i] = static_cast<float>(lonRad[i] * 180.0 / M_PI);
          latDeg[i] = static_cast<float>(latRad[i] * 180.0 / M_PI);

          // Normalize longitude to [0, 360) range
          if (lonDeg[i] < 0)
            lonDeg[i] += 360.0f;
        }

        // Create equirectangular grid (720x360 for 0.5 degree resolution)
        const size_t texWidth = 21600; // 0.5 degree longitude resolution
        const size_t texHeight = 10800; // 0.5 degree latitude resolution

        logInfo("[netcdf_utils] Creating equirectangular texture: %zux%zu",
            texWidth,
            texHeight);

        // Create 2D texture array
        array = ctx.createArray(ANARI_FLOAT32, texWidth, texHeight);
        float *texture = reinterpret_cast<float *>(array->map());

        // Initialize texture with fill value
        // const float fillValue = -9.e+33f; // From NetCDF fill value
        const float fillValue = 0.f; // From NetCDF fill value
        std::fill(texture, texture + (texWidth * texHeight), fillValue);

        // Map unstructured data to regular grid using nearest neighbor
        for (size_t i = 0; i < numCells; ++i) {
          // Skip invalid data
          if (cloudData[i] == fillValue || cloudData[i] != cloudData[i])
            continue; // NaN check

          // Convert coordinates to texture indices
          int texX = static_cast<int>((lonDeg[i] / 360.0f) * texWidth);
          int texY =
              static_cast<int>(((latDeg[i] + 90.0f) / 180.0f) * texHeight);

          // Clamp to valid range
          texX = std::max(0, std::min(static_cast<int>(texWidth - 1), texX));
          texY = std::max(0, std::min(static_cast<int>(texHeight - 1), texY));

          // Store data in texture (flip Y for proper orientation)
          size_t texIdx = (texHeight - 1 - texY) * texWidth + texX;
          texture[texIdx] = cloudData[i] * 1e6f;
        }

        array->unmap();

        logInfo(
            "[netcdf_utils] Successfully created equirectangular texture from %zu unstructured cells",
            numCells);

      } else {
        // Handle structured grid (original code path)
        size_t width = dims[0].getSize();
        size_t height = dims[1].getSize();
        size_t depth = dims[2].getSize();

        // Read height data from first slice (depth=0) of 3D data
        std::vector<float> heightData(width * height);
        std::vector<size_t> start = {0, 0, 0};
        std::vector<size_t> count = {width, height, 1};
        var.getVar(start, count, heightData.data());

        // Extract longitude and latitude coordinate information
        std::vector<float> lonData(width);
        std::vector<float> latData(height);
        bool hasCoordinates = false;

        try {
          // Try common coordinate variable names
          std::vector<std::string> lonNames = {"longitude", "lon", "x"};
          std::vector<std::string> latNames = {"latitude", "lat", "y"};

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
        array = ctx.createArray(ANARI_FLOAT32_VEC3, width, height);
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

    logInfo("[netcdf_utils] Successfully loaded '%s' as %s",
        paramName.c_str(),
        variableName.c_str());
    return array;
  } catch (const netCDF::exceptions::NcException &e) {
    logError("[netcdf_utils] NetCDF error loading '%s': %s",
        filepath.c_str(),
        e.what());
    return {};
  }
}

ArrayRef loadNetCDFSlice(Context &ctx,
    const std::string &filepath,
    const std::string &variableName,
    const std::string &paramName,
    size_t sliceIndex)
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
    std::vector<netCDF::NcDim> dims = var.getDims();
    if (dims.size() != 3) {
      logError("[netcdf_utils] Variable '%s' has %zu dimensions, expected 3",
          variableName.c_str(),
          dims.size());
      return {};
    }

    size_t width = dims[0].getSize();
    size_t height = dims[1].getSize();
    size_t depth = dims[2].getSize();

    if (sliceIndex >= depth) {
      logError("[netcdf_utils] Slice index %zu out of range (0-%zu)",
          sliceIndex,
          depth - 1);
      return {};
    }

    logInfo(
        "[netcdf_utils] Loading 2D slice %zu from 3D variable '%s': %zux%zu",
        sliceIndex,
        variableName.c_str(),
        width,
        height);

    // Create array for 2D slice
    ArrayRef array;
    nc_type dataType = var.getType().getId();

    if (dataType == NC_FLOAT) {
      array = ctx.createArray(ANARI_FLOAT32, width, height);
      float *data = reinterpret_cast<float *>(array->map());

      // Read specific slice
      std::vector<size_t> start = {0, 0, sliceIndex};
      std::vector<size_t> count = {width, height, 1};
      var.getVar(start, count, data);

      array->unmap();
    } else if (dataType == NC_DOUBLE) {
      array = ctx.createArray(ANARI_FLOAT32, width, height);
      float *data = reinterpret_cast<float *>(array->map());
      std::vector<double> tempData(width * height);

      // Read specific slice
      std::vector<size_t> start = {0, 0, sliceIndex};
      std::vector<size_t> count = {width, height, 1};
      var.getVar(start, count, tempData.data());

      // Convert double to float
      for (size_t i = 0; i < tempData.size(); ++i) {
        data[i] = static_cast<float>(tempData[i]);
      }
      array->unmap();
    } else if (dataType == NC_INT) {
      array = ctx.createArray(ANARI_FLOAT32, width, height);
      float *data = reinterpret_cast<float *>(array->map());
      std::vector<int> tempData(width * height);

      // Read specific slice
      std::vector<size_t> start = {0, 0, sliceIndex};
      std::vector<size_t> count = {width, height, 1};
      var.getVar(start, count, tempData.data());

      // Convert int to float
      for (size_t i = 0; i < tempData.size(); ++i) {
        data[i] = static_cast<float>(tempData[i]);
      }
      array->unmap();
    } else if (dataType == NC_SHORT) {
      array = ctx.createArray(ANARI_FLOAT32, width, height);
      float *data = reinterpret_cast<float *>(array->map());
      std::vector<short> tempData(width * height);

      // Read specific slice
      std::vector<size_t> start = {0, 0, sliceIndex};
      std::vector<size_t> count = {width, height, 1};
      var.getVar(start, count, tempData.data());

      // Convert short to float
      for (size_t i = 0; i < tempData.size(); ++i) {
        data[i] = static_cast<float>(tempData[i]);
      }
      array->unmap();
    } else {
      logError("[netcdf_utils] Unsupported data type %d for variable '%s'",
          dataType,
          variableName.c_str());
      return {};
    }

    logInfo("[netcdf_utils] Successfully loaded slice %zu of '%s' as %s",
        sliceIndex,
        paramName.c_str(),
        variableName.c_str());
    return array;

  } catch (const netCDF::exceptions::NcException &e) {
    logError("[netcdf_utils] NetCDF error loading slice from '%s': %s",
        filepath.c_str(),
        e.what());
    return {};
  }
}

#endif // TSD_USE_NETCDF

} // namespace tsd