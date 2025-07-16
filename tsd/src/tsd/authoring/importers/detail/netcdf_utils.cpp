// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "netcdf_utils.h"

#ifdef TSD_USE_NETCDF

#include <netcdf>
#include <algorithm>
#include <cstring>

namespace tsd {

NetCDFVariableInfo getNetCDFVariableInfo(const std::string& filepath, const std::string& variableName) {
  NetCDFVariableInfo info;
  
  try {
    netCDF::NcFile file(filepath, netCDF::NcFile::read);
    netCDF::NcVar var = file.getVar(variableName);
    
    if (var.isNull()) {
      logError("[netcdf_utils] Variable '%s' not found in file '%s'", variableName.c_str(), filepath.c_str());
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
            variableName.c_str(), info.dimensions.size(), info.totalElements);
    
  } catch (const netCDF::exceptions::NcException& e) {
    logError("[netcdf_utils] NetCDF error reading '%s': %s", filepath.c_str(), e.what());
  }
  
  return info;
}

ArrayRef loadNetCDFVariable(Context& ctx, 
                           const std::string& filepath, 
                           const std::string& variableName,
                           const std::string& paramName) {
  
  try {
    netCDF::NcFile file(filepath, netCDF::NcFile::read);
    netCDF::NcVar var = file.getVar(variableName);
    
    if (var.isNull()) {
      logError("[netcdf_utils] Variable '%s' not found in file '%s'", variableName.c_str(), filepath.c_str());
      return {};
    }
    
    // Get dimensions
    std::vector<netCDF::NcDim> dims = var.getDims();
    if (dims.size() != 3) {
      logError("[netcdf_utils] Variable '%s' has %zu dimensions, expected 3", 
               variableName.c_str(), dims.size());
      return {};
    }
    
    size_t width = dims[0].getSize();
    size_t height = dims[1].getSize();
    size_t depth = dims[2].getSize();
    
    logInfo("[netcdf_utils] Loading 3D variable '%s': %zux%zux%zu", 
            variableName.c_str(), width, height, depth);
    
    // Create array based on data type
    ArrayRef array;
    nc_type dataType = var.getType().getId();
    
    if (dataType == NC_FLOAT) {
      array = ctx.createArray(ANARI_FLOAT32, width, height, depth);
      float* data = reinterpret_cast<float*>(array->map());
      var.getVar(data);
      array->unmap();
    } else if (dataType == NC_DOUBLE) {
      array = ctx.createArray(ANARI_FLOAT32, width, height, depth);
      float* data = reinterpret_cast<float*>(array->map());
      std::vector<double> tempData(width * height * depth);
      var.getVar(tempData.data());
      
      // Convert double to float
      for (size_t i = 0; i < tempData.size(); ++i) {
        data[i] = static_cast<float>(tempData[i]);
      }
      array->unmap();
    } else if (dataType == NC_INT) {
      array = ctx.createArray(ANARI_FLOAT32, width, height, depth);
      float* data = reinterpret_cast<float*>(array->map());
      std::vector<int> tempData(width * height * depth);
      var.getVar(tempData.data());
      
      // Convert int to float
      for (size_t i = 0; i < tempData.size(); ++i) {
        data[i] = static_cast<float>(tempData[i]);
      }
      array->unmap();
    } else if (dataType == NC_SHORT) {
      array = ctx.createArray(ANARI_FLOAT32, width, height, depth);
      float* data = reinterpret_cast<float*>(array->map());
      std::vector<short> tempData(width * height * depth);
      var.getVar(tempData.data());
      
      // Convert short to float
      for (size_t i = 0; i < tempData.size(); ++i) {
        data[i] = static_cast<float>(tempData[i]);
      }
      array->unmap();
    } else {
      logError("[netcdf_utils] Unsupported data type %d for variable '%s'", dataType, variableName.c_str());
      return {};
    }
    
    logInfo("[netcdf_utils] Successfully loaded '%s' as %s", paramName.c_str(), variableName.c_str());
    return array;
    
  } catch (const netCDF::exceptions::NcException& e) {
    logError("[netcdf_utils] NetCDF error loading '%s': %s", filepath.c_str(), e.what());
    return {};
  }
}

ArrayRef loadNetCDFSlice(Context& ctx,
                        const std::string& filepath,
                        const std::string& variableName,
                        const std::string& paramName,
                        size_t sliceIndex) {
  
  try {
    netCDF::NcFile file(filepath, netCDF::NcFile::read);
    netCDF::NcVar var = file.getVar(variableName);
    
    if (var.isNull()) {
      logError("[netcdf_utils] Variable '%s' not found in file '%s'", variableName.c_str(), filepath.c_str());
      return {};
    }
    
    // Get dimensions
    std::vector<netCDF::NcDim> dims = var.getDims();
    if (dims.size() != 3) {
      logError("[netcdf_utils] Variable '%s' has %zu dimensions, expected 3", 
               variableName.c_str(), dims.size());
      return {};
    }
    
    size_t width = dims[0].getSize();
    size_t height = dims[1].getSize();
    size_t depth = dims[2].getSize();
    
    if (sliceIndex >= depth) {
      logError("[netcdf_utils] Slice index %zu out of range (0-%zu)", sliceIndex, depth - 1);
      return {};
    }
    
    logInfo("[netcdf_utils] Loading 2D slice %zu from 3D variable '%s': %zux%zu", 
            sliceIndex, variableName.c_str(), width, height);
    
    // Create array for 2D slice
    ArrayRef array;
    nc_type dataType = var.getType().getId();
    
    if (dataType == NC_FLOAT) {
      array = ctx.createArray(ANARI_FLOAT32, width, height);
      float* data = reinterpret_cast<float*>(array->map());
      
      // Read specific slice
      std::vector<size_t> start = {0, 0, sliceIndex};
      std::vector<size_t> count = {width, height, 1};
      var.getVar(start, count, data);
      
      array->unmap();
    } else if (dataType == NC_DOUBLE) {
      array = ctx.createArray(ANARI_FLOAT32, width, height);
      float* data = reinterpret_cast<float*>(array->map());
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
      float* data = reinterpret_cast<float*>(array->map());
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
      float* data = reinterpret_cast<float*>(array->map());
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
      logError("[netcdf_utils] Unsupported data type %d for variable '%s'", dataType, variableName.c_str());
      return {};
    }
    
    logInfo("[netcdf_utils] Successfully loaded slice %zu of '%s' as %s", 
            sliceIndex, paramName.c_str(), variableName.c_str());
    return array;
    
  } catch (const netCDF::exceptions::NcException& e) {
    logError("[netcdf_utils] NetCDF error loading slice from '%s': %s", filepath.c_str(), e.what());
    return {};
  }
}

#endif // TSD_USE_NETCDF

} // namespace tsd 