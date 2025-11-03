// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "import_AURORA.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
// std
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using namespace std;
namespace fs = filesystem;

namespace tsd::io {

struct AuroraHeader
{
  // Aurora parameters
  float intensity{1.0f};
  float waveFrequency{1.0f};
  float waveAmplitude{0.1f};
  float time{0.0f};
  float altitudeMin{100.0f};
  float altitudeMax{400.0f};
  float thickness{50.0f};
  float turbulence{0.5f};
  float numCurtains{5.0f};
  float magneticLatitude{65.0f};
  float unitDistance{1.0f};
  string colormap;
};

AuroraHeader readAuroraHeader(const string &filename)
{
  AuroraHeader header;
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

      // Remove 'f' suffix from float values if present
      if (!value.empty() && value.back() == 'f') {
        value.pop_back();
      }

      // Parse fields
      if (key == "intensity") {
        header.intensity = stof(value);
      } else if (key == "waveFrequency") {
        header.waveFrequency = stof(value);
      } else if (key == "waveAmplitude") {
        header.waveAmplitude = stof(value);
      } else if (key == "time") {
        header.time = stof(value);
      } else if (key == "altitudeMin") {
        header.altitudeMin = stof(value);
      } else if (key == "altitudeMax") {
        header.altitudeMax = stof(value);
      } else if (key == "thickness") {
        header.thickness = stof(value);
      } else if (key == "turbulence") {
        header.turbulence = stof(value);
      } else if (key == "numCurtains") {
        header.numCurtains = stof(value);
      } else if (key == "magneticLatitude") {
        header.magneticLatitude = stof(value);
      } else if (key == "unitDistance") {
        header.unitDistance = stof(value);
      } else if (key == "colormap") {
        header.colormap = value;
      }
    }
  }

  return header;
}

SpatialFieldRef import_AURORA(Scene &scene, const char *filepath)
{
  const auto header = readAuroraHeader(filepath);

  // Create a custom "Aurora" spatial field that will use raw ANARI calls
  auto field = scene.createObject<SpatialField>(tokens::spatial_field::aurora);
  field->setName(fileOf(filepath).c_str());

  // Log the parsed parameters
  logInfo("[import_Aurora] Aurora field parameters loaded:");
  logInfo("  intensity: %f", header.intensity);
  logInfo("  waveFrequency: %f", header.waveFrequency);
  logInfo("  waveAmplitude: %f", header.waveAmplitude);
  logInfo("  time: %f", header.time);
  logInfo("  altitudeMin: %f", header.altitudeMin);
  logInfo("  altitudeMax: %f", header.altitudeMax);
  logInfo("  thickness: %f", header.thickness);
  logInfo("  turbulence: %f", header.turbulence);
  logInfo("  numCurtains: %f", header.numCurtains);
  logInfo("  magneticLatitude: %f", header.magneticLatitude);
  logInfo("  unitDistance: %f", header.unitDistance);

  // Set Aurora-specific parameters directly on the TSD object
  // These will be used when the ANARI object is created
  field->setParameter("intensity", header.intensity);
  field->setParameter("waveFrequency", header.waveFrequency);
  field->setParameter("waveAmplitude", header.waveAmplitude);
  field->setParameter("time", header.time);
  field->setParameter("altitudeMin", header.altitudeMin);
  field->setParameter("altitudeMax", header.altitudeMax);
  field->setParameter("thickness", header.thickness);
  field->setParameter("turbulence", header.turbulence);
  field->setParameter("numCurtains", header.numCurtains);
  field->setParameter("magneticLatitude", header.magneticLatitude);

  // Set metadata
  field->setMetadataValue("unitDistance", header.unitDistance);

  return field;
}

std::string getTransferFunctionName_AURORA(const char *filename)
{
  const auto header = readAuroraHeader(filename);
  return header.colormap;
}

} // namespace tsd::io
