#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <learned_hashing.hpp>
#include <limits>
#include <tuple>
#include <vector>

#include "../support/datasets.hpp"

/// Tests whether RadixSpline is monotone for non-keys. This is important, e.g.,
/// if trained on a sample or used within a monotone Hashtable
TEST(RadixSpline, IsMonotoneForNonKeys) {
  using Data = std::uint64_t;

  // generate test datasets
  std::vector<std::vector<Data>> datasets{
      {1, 2, 4, 7, 10, 1000},
      dataset::load_cached(dataset::ID::GAPPED_10, 10000)};

  for (const auto& dataset : datasets) {
    // build monotone rmi model
    const learned_hashing::RadixSplineHash<Data> rs(
        dataset.begin(), dataset.end(), dataset.size());

    // test monotony
    size_t last_i = 0;
    for (Data k = *std::min_element(dataset.begin(), dataset.end());
         k < *std::max_element(dataset.begin(), dataset.end()); k++) {
      size_t new_i = rs(k);
      EXPECT_GE(new_i, last_i);
      last_i = new_i;
    }
  }
}
