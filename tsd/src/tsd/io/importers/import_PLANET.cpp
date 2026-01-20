// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/Logging.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
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

namespace tsd::io {

struct PlanetHeader
{
  // Planet parameters
  float planetRadius{0.5f};
  float elevationScale{0.1f};
  float unitDistance{256.0f};

  // Map file paths
  string elevationMapPath;
  string diffuseMapPath;
  string normalMapPath;
  string colormap;
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
      if (key == "planetRadius") {
        header.planetRadius = stof(value);
      } else if (key == "elevationScale") {
        header.elevationScale = stof(value);
      } else if (key == "unitDistance") {
        header.unitDistance = stof(value);
      } else if (key == "elevationMap") {
        header.elevationMapPath = value;
      } else if (key == "diffuseMap") {
        header.diffuseMapPath = value;
      } else if (key == "normalMap") {
        header.normalMapPath = value;
      } else if (key == "colormap") {
        header.colormap = value;
      }
    }
  }

  return header;
}

// Helper function to load image data into array buffer
bool loadImageData(Scene &scene,
    SpatialFieldRef field,
    const string &filePath,
    const string &paramName,
    bool required = false)
{
  // Load image using stb_image
  int width, height, channels;
  // Don't flip equirectangular textures - they should match NetCDF coordinate convention
  stbi_set_flip_vertically_on_load(0);
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
    imageArray = scene.createArray(ANARI_FLOAT32, width, height);
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
    imageArray = scene.createArray(ANARI_FLOAT32_VEC4, width, height);
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
  } else if (paramName == "diffuseMap") {
    // Special handling for diffuse map: create colormap and index image
    // Quantize colors to reduce unique colors and limit to 16-bit max
    const uint32_t MAX_COLORS = 65536; // 16-bit max
    std::map<std::vector<unsigned char>, uint32_t> colorToIndex;
    std::vector<std::vector<unsigned char>> colormap;

    // Quantization factor: reduce 8-bit values to fewer bits to limit unique
    // colors For RGB: use 5 bits per channel (32 levels each) = 32^3 = 32,768
    // colors max For RGBA: use 4 bits per channel (16 levels each) = 16^4 =
    // 65,536 colors max
    int quantizationBits = (channels == 4) ? 4 : 5;
    int quantizationShift = 8 - quantizationBits;
    int quantizationLevels = 1 << quantizationBits;

    // First pass: collect quantized unique colors
    for (int i = 0; i < width * height; ++i) {
      std::vector<unsigned char> quantizedColor;
      for (int c = 0; c < channels; ++c) {
        // Quantize the color value
        unsigned char originalValue = imageData[i * channels + c];
        unsigned char quantizedValue = (originalValue >> quantizationShift)
            << quantizationShift;
        quantizedColor.push_back(quantizedValue);
      }

      if (colorToIndex.find(quantizedColor) == colorToIndex.end()) {
        // Check if we've reached the maximum number of colors
        if (colormap.size() >= MAX_COLORS) {
          logWarning(
              "[import_Planet] Reached maximum colormap size (%u), truncating",
              MAX_COLORS);
          break;
        }
        colorToIndex[quantizedColor] = static_cast<uint32_t>(colormap.size());
        colormap.push_back(quantizedColor);
      }
    }

    // Create colormap array
    ArrayRef colormapArray;
    if (channels == 1) {
      colormapArray = scene.createArray(
          ANARI_UFIXED8, static_cast<uint32_t>(colormap.size()));
    } else if (channels == 3) {
      colormapArray = scene.createArray(
          ANARI_UFIXED8_VEC3, static_cast<uint32_t>(colormap.size()));
    } else if (channels == 4) {
      colormapArray = scene.createArray(
          ANARI_UFIXED8_VEC4, static_cast<uint32_t>(colormap.size()));
    } else {
      logWarning(
          "[import_Planet] Unsupported channel count %d for diffuse map, using single channel",
          channels);
      colormapArray = scene.createArray(
          ANARI_UFIXED8, static_cast<uint32_t>(colormap.size()));
    }

    // Fill colormap array
    std::vector<unsigned char> colormapData;
    colormapData.reserve(colormap.size() * channels);
    for (size_t i = 0; i < colormap.size(); ++i) {
      for (int c = 0; c < channels; ++c) {
        colormapData.push_back(colormap[i][c]);
      }
    }
    colormapArray->setData(colormapData.data());

    // Create normalized index image array as floats
    imageArray = scene.createArray(ANARI_FLOAT32, width, height);
    std::vector<float> indexData;
    indexData.reserve(width * height);

    // Second pass: create normalized index image using quantized colors
    float colormapSizeFloat = static_cast<float>(colormap.size());
    for (int i = 0; i < width * height; ++i) {
      std::vector<unsigned char> quantizedColor;
      for (int c = 0; c < channels; ++c) {
        // Quantize the color value (same as first pass)
        unsigned char originalValue = imageData[i * channels + c];
        unsigned char quantizedValue = (originalValue >> quantizationShift)
            << quantizationShift;
        quantizedColor.push_back(quantizedValue);
      }
      // Normalize the index to [0, 1] range
      float normalizedIndex =
          static_cast<float>(colorToIndex[quantizedColor]) / colormapSizeFloat;
      indexData.push_back(normalizedIndex);
    }
    imageArray->setData(indexData.data());

    // Store colormap data in field metadata for import_volume to use
    field->setMetadataValue("hasColormap", false);
    field->setMetadataValue("colormapSize", static_cast<int>(colormap.size()));
    field->setMetadataValue("colormapChannels", channels);

    // Store the colormap array in metadata (this will be accessed by
    // import_volume)
    field->setParameterObject("_colormap", *colormapArray);
    field->setParameterObject("_indices", *imageArray);

    logInfo("  Loaded %s: %dx%d, %d channels, %zu quantized colors (max %u)",
        paramName.c_str(),
        width,
        height,
        channels,
        colormap.size(),
        MAX_COLORS);
    logInfo("  Quantization: %d bits per channel (%d levels), %d-bit shift",
        quantizationBits,
        quantizationLevels,
        quantizationShift);
    logInfo(
        "  Stored colormap in field metadata with %zu colors", colormap.size());
    logInfo(
        "  Stored normalized float indices with %dx%d indices", width, height);
  } else {
    // For other maps, keep as uint8
    if (channels == 1) {
      imageArray = scene.createArray(ANARI_UFIXED8, width, height);
    } else if (channels == 3) {
      imageArray = scene.createArray(ANARI_UFIXED8_VEC3, width, height);
    } else if (channels == 4) {
      imageArray = scene.createArray(ANARI_UFIXED8_VEC4, width, height);
    } else {
      logWarning(
          "[import_Planet] Unsupported channel count %d for %s, using single channel",
          channels,
          paramName.c_str());
      imageArray = scene.createArray(ANARI_UFIXED8, width, height);
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

SpatialFieldRef import_PLANET(Scene &scene, const char *filepath)
{
  const auto header = readPlanetHeader(filepath);
  const auto basePath = fs::path(filepath).parent_path().string();

  // Create a custom "Planet" spatial field that will use raw ANARI calls
  auto field = scene.createObject<SpatialField>(tokens::spatial_field::planet);
  field->setName(fileOf(filepath).c_str());

  // Log the parsed parameters
  logInfo("[import_Planet] Planet parameters loaded:");
  logInfo("  PlanetRadius: %f", header.planetRadius);
  logInfo("  elevationScale: %f", header.elevationScale);

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
  field->setParameter("planetRadius", header.planetRadius);
  field->setParameter("elevationScale", header.elevationScale);

  // Load elevation map (mandatory)
  if (header.elevationMapPath.empty()) {
    logError("[import_Planet] Elevation map is required but not provided");
    return {};
  }

  string fullElevationPath = basePath + "/" + header.elevationMapPath;
  if (!loadImageData(scene, field, fullElevationPath, "elevationMap", true)) {
    return {};
  }

  // Load diffuse map (optional)
  if (!header.diffuseMapPath.empty()) {
    string fullDiffusePath = basePath + "/" + header.diffuseMapPath;
    loadImageData(scene, field, fullDiffusePath, "diffuseMap", false);
  }

  // Load normal map (optional)
  if (!header.normalMapPath.empty()) {
    string fullNormalPath = basePath + "/" + header.normalMapPath;
    loadImageData(scene, field, fullNormalPath, "normalMap", false);
  }

  field->setMetadataValue("unitDistance", header.unitDistance);

  return field;
}

std::string getTransferFunctionName_PLANET(const char *filename)
{
  const auto header = readPlanetHeader(filename);
  return header.colormap;
}

} // namespace tsd::io
