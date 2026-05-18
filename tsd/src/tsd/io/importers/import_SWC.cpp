// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/Logging.hpp"
#include "tsd/io/importers.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <vector>

namespace {
const float DEFAULT_ROUGHNESS = 0.5f;
const float DEFAULT_METALLIC = 0.5f;
} // namespace

namespace tsd::io {

/**
 * Represents a point in a SWC (Standard Warehouse Connector) file.
 */
struct SWCPoint
{
  int id;
  int type;
  double x, y, z;
  double radius;
  int parent;
};

// ---------------------------------------------------------------------------
// SDF primitive layout — must match SDFPrimitive in devices/rtx/device/gpu/gpu_objects.h
// byte-for-byte. The static_assert below guards against drift.
// ---------------------------------------------------------------------------

enum class SDFType : uint8_t
{
  SPHERE = 0,
  PILL = 1,
  CONE_PILL = 2,
  CONE_PILL_SIGMOID = 3,
  CONE = 4,
  TORUS = 5,
  CUT_SPHERE = 6,
  VESICA = 7,
  ELLIPSOID = 8
};

struct SDFPrimLayout
{
  uint64_t userData{0};
  float    userParams[3]{0.f, 0.f, 0.f};
  float    p0[3]{0.f, 0.f, 0.f};
  float    p1[3]{0.f, 0.f, 0.f};
  float    r0{-1.f};
  float    r1{-1.f};
  uint32_t _pad{0};
  uint64_t neighboursIndex{0};
  uint8_t  numNeighbours{0};
  uint8_t  type{0};
  uint8_t  _pad2[6]{};
};

static_assert(sizeof(SDFPrimLayout) == 72,
    "SDFPrimLayout size must be 72 bytes to match SDFPrimitive in gpu_objects.h");
static_assert(offsetof(SDFPrimLayout, neighboursIndex) == 56,
    "SDFPrimLayout::neighboursIndex offset mismatch");

// ---------------------------------------------------------------------------

/**
 * Builds an SDF geometry from a parsed SWC skeleton.
 *
 * Each SWC point becomes one SDF primitive:
 *  - Root points (parent == -1) → SPHERE
 *  - All other points           → CONE_PILL from the point to its parent
 *
 * Neighbour connections are established so that smooth-min blending fires at
 * every junction (branch points, connection to parent segment, connection to
 * child segments), giving organic surface continuity across the morphology.
 */
void readSWCFile(
    Scene &scene, const std::string &filename, LayerNodeRef location)
{
  std::ifstream file(filename);
  if (!file.is_open()) {
    logError("[import_SWC] Error opening file: %s", filename.c_str());
    return;
  }

  std::map<int, SWCPoint> points;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::istringstream iss(line);
    SWCPoint pt;
    if (iss >> pt.id >> pt.type >> pt.x >> pt.y >> pt.z >> pt.radius
        >> pt.parent)
      points[pt.id] = pt;
  }
  file.close();

  if (points.empty()) {
    logWarning("[import_SWC] No points found in %s", filename.c_str());
    return;
  }

  if (!location)
    location = scene.defaultLayer()->root();

  // ------------------------------------------------------------------
  // Step 1: build one SDF primitive per SWC point and record the mapping
  //         SWC-point-ID -> SDF-buffer-index.
  // ------------------------------------------------------------------
  std::map<int, size_t> pointToSdfIdx;

  struct PrimBuild
  {
    SDFPrimLayout prim;
    int swcId{-1};
    int parentId{-1};
  };

  std::vector<PrimBuild> primBuilds;
  primBuilds.reserve(points.size());

  for (const auto &[id, pt] : points) {
    PrimBuild build;
    build.swcId = id;
    build.parentId = pt.parent;

    build.prim.userData = static_cast<uint64_t>(id);

    if (pt.parent == -1) {
      // Root → sphere
      build.prim.p0[0] = static_cast<float>(pt.x);
      build.prim.p0[1] = static_cast<float>(pt.y);
      build.prim.p0[2] = static_cast<float>(pt.z);
      build.prim.r0 = static_cast<float>(pt.radius);
      build.prim.type = static_cast<uint8_t>(SDFType::SPHERE);
    } else {
      // Non-root → CONE_PILL from this point to its parent
      const auto &par = points.at(pt.parent);

      float px = static_cast<float>(pt.x);
      float py = static_cast<float>(pt.y);
      float pz = static_cast<float>(pt.z);
      float qx = static_cast<float>(par.x);
      float qy = static_cast<float>(par.y);
      float qz = static_cast<float>(par.z);
      float r0 = static_cast<float>(pt.radius);
      float r1 = static_cast<float>(par.radius);

      // ConePill requires the larger radius at p0
      if (r0 < r1) {
        std::swap(px, qx);
        std::swap(py, qy);
        std::swap(pz, qz);
        std::swap(r0, r1);
      }

      build.prim.p0[0] = px; build.prim.p0[1] = py; build.prim.p0[2] = pz;
      build.prim.p1[0] = qx; build.prim.p1[1] = qy; build.prim.p1[2] = qz;
      build.prim.r0 = r0;
      build.prim.r1 = r1;
      build.prim.type = static_cast<uint8_t>(SDFType::CONE_PILL);
    }

    pointToSdfIdx[id] = primBuilds.size();
    primBuilds.push_back(std::move(build));
  }

  // ------------------------------------------------------------------
  // Step 2: build the children map for quick lookup of SWC adjacency.
  // ------------------------------------------------------------------
  std::map<int, std::vector<int>> swcChildren;
  for (const auto &[id, pt] : points)
    if (pt.parent != -1)
      swcChildren[pt.parent].push_back(id);

  // ------------------------------------------------------------------
  // Step 3: for each primitive, collect its neighbours and fill the flat
  //         neighbour buffer.  Neighbours are all SDF primitives that share
  //         an endpoint with the current primitive:
  //
  //   • child segments   (primitives for direct SWC children)
  //   • parent primitive (primitive for the SWC parent, sphere or segment)
  //   • sibling segments (other SWC children of the same parent)
  //
  // This covers every junction type: tips, straight runs, and branch points.
  // ------------------------------------------------------------------
  std::vector<uint64_t> neighbourBuffer;

  for (auto &build : primBuilds) {
    const int myId = build.swcId;
    const int parentId = build.parentId;

    std::vector<uint64_t> nbrs;

    // Children of this point
    auto childIt = swcChildren.find(myId);
    if (childIt != swcChildren.end()) {
      for (int childId : childIt->second)
        nbrs.push_back(static_cast<uint64_t>(pointToSdfIdx.at(childId)));
    }

    if (parentId != -1) {
      // The primitive representing the parent point
      nbrs.push_back(static_cast<uint64_t>(pointToSdfIdx.at(parentId)));

      // Siblings: other children of the same parent
      auto sibIt = swcChildren.find(parentId);
      if (sibIt != swcChildren.end()) {
        for (int sibId : sibIt->second) {
          if (sibId != myId)
            nbrs.push_back(static_cast<uint64_t>(pointToSdfIdx.at(sibId)));
        }
      }
    }

    // Deduplicate and cap at 255
    std::sort(nbrs.begin(), nbrs.end());
    nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
    // Remove self-reference if somehow present
    nbrs.erase(
        std::remove(nbrs.begin(), nbrs.end(),
            static_cast<uint64_t>(pointToSdfIdx.at(myId))),
        nbrs.end());

    const uint8_t numNbrs =
        static_cast<uint8_t>(std::min(nbrs.size(), static_cast<size_t>(255)));

    build.prim.neighboursIndex = static_cast<uint64_t>(neighbourBuffer.size());
    build.prim.numNeighbours = numNbrs;

    for (uint8_t i = 0; i < numNbrs; i++)
      neighbourBuffer.push_back(nbrs[i]);
  }

  // ------------------------------------------------------------------
  // Step 4: pack into flat byte arrays and upload to the TSD scene.
  // ------------------------------------------------------------------
  const size_t numPrims = primBuilds.size();

  std::vector<uint8_t> sdfRawBytes(numPrims * sizeof(SDFPrimLayout));
  for (size_t i = 0; i < numPrims; i++) {
    std::memcpy(sdfRawBytes.data() + i * sizeof(SDFPrimLayout),
        &primBuilds[i].prim,
        sizeof(SDFPrimLayout));
  }

  // primitive.sdf  : raw bytes, element type UINT8
  auto sdfArray = scene.createArray(ANARI_UINT8, sdfRawBytes.size());
  sdfArray->setData(sdfRawBytes);

  // primitive.neighbor : flat uint64 index buffer
  ArrayRef neighbourArray;
  if (!neighbourBuffer.empty()) {
    neighbourArray = scene.createArray(ANARI_UINT64, neighbourBuffer.size());
    neighbourArray->setData(neighbourBuffer);
  }

  logInfo("[import_SWC] Built %zu SDF primitives, %zu neighbour entries from %s",
      numPrims,
      neighbourBuffer.size(),
      filename.c_str());

  // ------------------------------------------------------------------
  // Step 5: create the SDF geometry object and assign parameters.
  // ------------------------------------------------------------------
  auto sdfGeom = scene.createObject<Geometry>(tokens::geometry::sdfGeometries);
  sdfGeom->setName("sdf_geometry");

  sdfGeom->setParameterObject("primitive.sdf", *sdfArray);
  if (neighbourArray)
    sdfGeom->setParameterObject("primitive.neighbor", *neighbourArray);

  const float epsilon = 1e-5f;
  const uint32_t marchIter = 128u;
  const float blendFactor = 1.f;
  const float blendLerpFactor = 0.5f;
  const float omega = 1.f;
  const float noiseFactor = 0.f;

  sdfGeom->setParameter("epsilon", ANARI_FLOAT32, &epsilon);
  sdfGeom->setParameter("nbMarchIterations", ANARI_UINT32, &marchIter);
  sdfGeom->setParameter("blendFactor", ANARI_FLOAT32, &blendFactor);
  sdfGeom->setParameter("blendLerpFactor", ANARI_FLOAT32, &blendLerpFactor);
  sdfGeom->setParameter("omega", ANARI_FLOAT32, &omega);
  sdfGeom->setParameter("noiseFactor", ANARI_FLOAT32, &noiseFactor);

  // ------------------------------------------------------------------
  // Step 6: material and scene graph.
  // ------------------------------------------------------------------
  auto m = scene.createObject<Material>(tokens::material::physicallyBased);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 0.5);
  tsd::math::float3 baseColor(
      0.5f + static_cast<float>(dis(gen)),
      0.5f + static_cast<float>(dis(gen)),
      0.5f + static_cast<float>(dis(gen)));

  const float metallic = DEFAULT_METALLIC;
  const float roughness = DEFAULT_ROUGHNESS;

  m->setParameter("baseColor", ANARI_FLOAT32_VEC3, &baseColor);
  m->setParameter("metallic", ANARI_FLOAT32, &metallic);
  m->setParameter("roughness", ANARI_FLOAT32, &roughness);

  const std::string basename =
      std::filesystem::path(filename).filename().string();

  const auto swcLocation = scene.insertChildNode(location, basename.c_str());

  auto surface = scene.createSurface(basename.c_str(), sdfGeom, m);
  scene.insertChildObjectNode(swcLocation, surface);
}

/**
 * Imports a single SWC file into the current context.
 */
void import_SWC(Scene &scene,
    tsd::animation::AnimationManager &animMgr,
    const char *filename,
    LayerNodeRef location)
{
  (void)animMgr;
  readSWCFile(scene, filename, location);
}

} // namespace tsd::io
