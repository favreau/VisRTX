// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/authoring/importers.hpp"
#include "tsd/authoring/importers/detail/importer_common.hpp"
#include "tsd/core/Logging.hpp"
// stb_image
#include "stb_image.h"
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

struct PlanetHeader
{
  // Planet parameters
  float PlanetRadius{6378000.0f};
  float sphereRadius{0.5f};
  float elevationScale{0.1f};
  float atmosphereThickness{0.2f};
  float3 sphereCenter{0.0f, 0.0f, 0.0f};

  // Map file paths
  string elevationMapPath;
  string diffuseMapPath;
  string normalMapPath;
};

PlanetHeader readPlanetHeader(const string &filename)
{
  PlanetHeader header;
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
      if (key == "PlanetRadius") {
        header.PlanetRadius = stof(value);
      } else if (key == "sphereRadius") {
        header.sphereRadius = stof(value);
      } else if (key == "elevationScale") {
        header.elevationScale = stof(value);
      } else if (key == "atmosphereThickness") {
        header.atmosphereThickness = stof(value);
      } else if (key == "sphereCenter") {
        sscanf(value.c_str(),
            "{%f, %f, %f}",
            &header.sphereCenter.x,
            &header.sphereCenter.y,
            &header.sphereCenter.z);
      } else if (key == "elevationMap") {
        header.elevationMapPath = value;
      } else if (key == "diffuseMap") {
        header.diffuseMapPath = value;
      } else if (key == "normalMap") {
        header.normalMapPath = value;
      }
    }
  }

  return header;
}

// Helper function to load image data into array buffer
bool loadImageData(Context &ctx,
    SpatialFieldRef field,
    const string &filePath,
    const string &paramName,
    bool required = false)
{
  // Load image using stb_image
  int width, height, channels;
  stbi_set_flip_vertically_on_load(1);
  unsigned char *imageData =
      stbi_load(filePath.c_str(), &width, &height, &channels, 0);

  if (!imageData) {
    if (required) {
      logError("[import_Planet] Failed to load required %s: %s",
          paramName.c_str(),
          filePath.c_str());
    } else {
      logWarning("[import_Planet] Failed to load %s: %s",
          paramName.c_str(),
          filePath.c_str());
    }
    return false;
  }

  // Create array for image data based on channels
  ArrayRef imageArray;

  // Special handling for elevation map: convert RGB uint8 to float
  if (paramName == "elevationMap") {
    imageArray = ctx.createArray(ANARI_FLOAT32, width, height);
    float *floatData = reinterpret_cast<float *>(imageArray->map());

    // Convert uint8 RGB values to float (0-255 -> 0.0-255.0)
    // Since all RGB values are identical, just use the first channel
    for (int i = 0; i < width * height; ++i) {
      floatData[i] = static_cast<float>(
          imageData[i * channels + 1] / 255.0f); // Take R channel
    }

    imageArray->unmap();
    logInfo("  Loaded %s: %dx%d, converted uint8 to float (0-255)",
        paramName.c_str(),
        width,
        height);
  } else if (paramName == "normalMap") {
    // Special handling for normal map: convert to vector of 4 floats
    imageArray = ctx.createArray(ANARI_FLOAT32_VEC4, width, height);
    float4 *float4Data = reinterpret_cast<float4 *>(imageArray->map());

    // Convert uint8 RGB(A) values to float4 (0-255 -> 0.0-1.0)
    for (int i = 0; i < width * height; ++i) {
      float4 normal{0.f, 0.f, 0.f, 1.f};
      normal.x =
          static_cast<float>(imageData[i * channels + 0]) / 255.0f; // R -> X
      normal.y =
          static_cast<float>(imageData[i * channels + 1]) / 255.0f; // G -> Y

      if (channels >= 3) {
        normal.z =
            static_cast<float>(imageData[i * channels + 2]) / 255.0f; // B -> Z
      }

      if (channels >= 4) {
        normal.w =
            static_cast<float>(imageData[i * channels + 3]) / 255.0f; // A -> W
      }

      float4Data[i] = normalize(0.5f * normal - 0.5f);
    }

    imageArray->unmap();
    logInfo("  Loaded %s: %dx%d, converted uint8 to float4",
        paramName.c_str(),
        width,
        height);
  } else {
    // For other maps (diffuse), keep as uint8
    if (channels == 1) {
      imageArray = ctx.createArray(ANARI_UFIXED8, width, height);
    } else if (channels == 3) {
      imageArray = ctx.createArray(ANARI_UFIXED8_VEC3, width, height);
    } else if (channels == 4) {
      imageArray = ctx.createArray(ANARI_UFIXED8_VEC4, width, height);
    } else {
      logWarning(
          "[import_Planet] Unsupported channel count %d for %s, using single channel",
          channels,
          paramName.c_str());
      imageArray = ctx.createArray(ANARI_UFIXED8, width, height);
    }

    // Copy image data to array
    size_t dataSize = width * height * channels;
    std::memcpy(imageArray->map(), imageData, dataSize);
    imageArray->unmap();

    logInfo("  Loaded %s: %dx%d, %d channels",
        paramName.c_str(),
        width,
        height,
        channels);
  }

  // Free stb_image data
  stbi_image_free(imageData);

  field->setParameterObject(paramName, *imageArray);
  return true;
}

SpatialFieldRef import_Planet(Context &ctx, const char *filepath)
{
  const auto header = readPlanetHeader(filepath);
  const auto basePath = fs::path(filepath).parent_path().string();

  // Create a custom "Planet" spatial field that will use raw ANARI calls
  auto field = ctx.createObject<SpatialField>(tokens::spatial_field::planet);
  field->setName(fileOf(filepath).c_str());

  // Log the parsed parameters
  logInfo("[import_Planet] Planet parameters loaded:");
  logInfo("  PlanetRadius: %f", header.PlanetRadius);
  logInfo("  sphereRadius: %f", header.sphereRadius);
  logInfo("  elevationScale: %f", header.elevationScale);
  logInfo("  atmosphereThickness: %f", header.atmosphereThickness);
  logInfo("  sphereCenter: (%f, %f, %f)",
      header.sphereCenter.x,
      header.sphereCenter.y,
      header.sphereCenter.z);

  if (!header.elevationMapPath.empty()) {
    logInfo("  elevationMap: %s", header.elevationMapPath.c_str());
  }
  if (!header.diffuseMapPath.empty()) {
    logInfo("  diffuseMap: %s", header.diffuseMapPath.c_str());
  }
  if (!header.normalMapPath.empty()) {
    logInfo("  normalMap: %s", header.normalMapPath.c_str());
  }

  // Set Planet-specific parameters directly on the TSD object
  // These will be used when the ANARI object is created
  field->setParameter("PlanetRadius", header.PlanetRadius);
  field->setParameter("sphereRadius", header.sphereRadius);
  field->setParameter("elevationScale", header.elevationScale);
  field->setParameter("atmosphereThickness", header.atmosphereThickness);
  field->setParameter("sphereCenter", header.sphereCenter);

  // Load elevation map (mandatory)
  if (header.elevationMapPath.empty()) {
    logError("[import_Planet] Elevation map is required but not provided");
    return {};
  }

  string fullElevationPath = basePath + "/" + header.elevationMapPath;
  if (!loadImageData(ctx, field, fullElevationPath, "elevationMap", true)) {
    return {};
  }

  // Load diffuse map (optional)
  if (!header.diffuseMapPath.empty()) {
    string fullDiffusePath = basePath + "/" + header.diffuseMapPath;
    loadImageData(ctx, field, fullDiffusePath, "diffuseMap", false);
  }

  // Load normal map (optional)
  if (!header.normalMapPath.empty()) {
    string fullNormalPath = basePath + "/" + header.normalMapPath;
    loadImageData(ctx, field, fullNormalPath, "normalMap", false);
  }

  return field;
}

} // namespace tsd
