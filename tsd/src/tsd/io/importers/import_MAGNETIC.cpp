// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "import_MAGNETIC.hpp"
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

struct MagneticHeader
{
  // Magnetic field parameters
  float equatorStrength{30.0f};
  float poleStrength{70.0f};
  float dipoleTilt{11.5f};
  tsd::math::float3 scale{1.0f, 1.0f, 1.0f};
  float unitDistance{1.0f};
  string colormap;
};

MagneticHeader readMagneticHeader(const string &filename)
{
  MagneticHeader header;
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
      if (key == "equatorStrength") {
        header.equatorStrength = stof(value);
      } else if (key == "poleStrength") {
        header.poleStrength = stof(value);
      } else if (key == "dipoleTilt") {
        header.dipoleTilt = stof(value);
      } else if (key == "scale") {
        // Parse as three floats: x y z
        std::istringstream iss(value);
        float x, y, z;
        if (iss >> x >> y >> z) {
          header.scale = tsd::math::float3(x, y, z);
        } else if (iss.clear(), iss.seekg(0), iss >> x) {
          // Single value - apply to all axes
          header.scale = tsd::math::float3(x, x, x);
        }
      } else if (key == "unitDistance") {
        header.unitDistance = stof(value);
      } else if (key == "colormap") {
        header.colormap = value;
      }
    }
  }

  return header;
}

SpatialFieldRef import_MAGNETIC(Scene &scene, const char *filepath)
{
  const auto header = readMagneticHeader(filepath);

  // Create a custom "Magnetic" spatial field that will use raw ANARI calls
  auto field =
      scene.createObject<SpatialField>(tokens::spatial_field::magnetic);
  field->setName(fileOf(filepath).c_str());

  // Log the parsed parameters
  logInfo("[import_Magnetic] Magnetic field parameters loaded:");
  logInfo("  equatorStrength: %f", header.equatorStrength);
  logInfo("  poleStrength: %f", header.poleStrength);
  logInfo("  dipoleTilt: %f", header.dipoleTilt);
  logInfo(
      "  scale: (%f, %f, %f)", header.scale.x, header.scale.y, header.scale.z);
  logInfo("  unitDistance: %f", header.unitDistance);

  // Set Magnetic-specific parameters directly on the TSD object
  // These will be used when the ANARI object is created
  field->setParameter("equatorStrength", header.equatorStrength);
  field->setParameter("poleStrength", header.poleStrength);
  field->setParameter("dipoleTilt", header.dipoleTilt);

  // Store scaling factor as metadata for volume transformation
  field->setMetadataValue("scale", header.scale);

  // Set metadata
  field->setMetadataValue("unitDistance", header.unitDistance);

  return field;
}

std::string getTransferFunctionName_MAGNETIC(const char *filename)
{
  const auto header = readMagneticHeader(filename);
  return header.colormap;
}

} // namespace tsd::io