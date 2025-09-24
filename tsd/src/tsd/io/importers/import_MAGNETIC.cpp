// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

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

  // Set Magnetic-specific parameters directly on the TSD object
  // These will be used when the ANARI object is created
  field->setParameter("equatorStrength", header.equatorStrength);
  field->setParameter("poleStrength", header.poleStrength);
  field->setParameter("dipoleTilt", header.dipoleTilt);

  // Set default metadata
  field->setMetadataValue("unitDistance", 1.f);

  return field;
}

} // namespace tsd::io