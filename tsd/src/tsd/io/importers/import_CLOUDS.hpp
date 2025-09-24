// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace tsd::io {

// Helper function to extract colormap name from CLOUDS header
std::string getTransferFunctionName_CLOUDS(const char *filename);

} // namespace tsd::io
