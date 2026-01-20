## Copyright 2024-2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsdDemoPlanet LANGUAGES CXX)

add_executable(${PROJECT_NAME}
  tsdDemoPlanet.cpp
  PlanetControls.cpp
  PlanetControls.h
)

target_link_libraries(${PROJECT_NAME}
PRIVATE
  tsd_ui_imgui
)
