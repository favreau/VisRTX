// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "RenderServer.hpp"

int main(int argc, const char *argv[])
{
  tsd::network::RenderServer server(argc, argv);
  server.run(49000);
  return 0;
}
