// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Logging.hpp"
#include "tsd/core/Context.hpp"
#include "tsd/objects/Array.hpp"
#include "tsd/objects/SpatialField.hpp"

#ifdef TSD_USE_NETCDF
#include <netcdf>
#endif

namespace tsd {

#ifdef TSD_USE_NETCDF

struct NetCDFVariableInfo {
  std::string name;
  std::vector<size_t> dimensions;
  std::vector<std::string> dimNames;
  nc_type dataType;
  size_t totalElements;
};

// Get information about a variable in a netCDF file
NetCDFVariableInfo getNetCDFVariableInfo(const std::string& filepath, const std::string& variableName);

// Load a 3D variable from netCDF file into an ANARI array
ArrayRef loadNetCDFVariable(Context& ctx, 
                           const std::string& filepath, 
                           const std::string& variableName,
                           const std::string& paramName);

// Load a 2D slice from a 3D netCDF variable
ArrayRef loadNetCDFSlice(Context& ctx,
                        const std::string& filepath,
                        const std::string& variableName,
                        const std::string& paramName,
                        size_t sliceIndex = 0);

#endif // TSD_USE_NETCDF

} // namespace tsd 