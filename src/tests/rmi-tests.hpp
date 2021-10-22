#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <learned_hashing.hpp>
#include <limits>
#include <tuple>
#include <vector>

#include "../support/datasets.hpp"

template <class Data, size_t I = 0, typename... Tp>
void iter_rmis(const std::tuple<Tp...>& t, const std::vector<Data>& dataset,
               const size_t dataset_size) {
  if constexpr (I + 1 != sizeof...(Tp))
    iter_rmis<Data, I + 1>(t, dataset, dataset_size);
}

// ==== RMI ====

// on sequential data, there mustn't be any collisions in theory.
// However, floating point imprecisions lead to (few!) collisions
// in practice
TEST(RMI, NoCollisionsOnSequential) {
  using Data = std::uint64_t;
  for (const auto dataset_size : {1000, 10000, 1000000}) {
    std::vector<Data> dataset(dataset_size, 0);
    for (size_t i = 0; i < dataset.size(); i++) dataset[i] = 20000 + i;

    const learned_hashing::RMIHash<Data, 100> rmi(dataset.begin(),
                                                  dataset.end(), dataset_size);

    size_t incidents = 0;
    std::vector<bool> slot_occupied(dataset_size, false);
    for (size_t i = 0; i < dataset.size(); i++) {
      const size_t index = rmi(dataset[i]);

      EXPECT_GE(index, 0);
      EXPECT_LT(index, dataset.size());

      incidents += slot_occupied[index];
      slot_occupied[index] = true;
    }
    EXPECT_LE(incidents, dataset_size / 100);
  }
}

TEST(RMI, ConstructionAlgorithmsMatch) {
  using Data = std::uint64_t;

  for (const auto dataset_size : {1000, 10000, 1000000}) {
    for (const auto did : {dataset::ID::SEQUENTIAL, dataset::ID::UNIFORM,
                           dataset::ID::UNIFORM, dataset::ID::GAPPED_10}) {
      const auto dataset = dataset::load_cached(did, dataset_size);

      const learned_hashing::RMIHash<Data, 10000> old_rmi(
          dataset.begin(), dataset.end(), dataset_size, false);
      const learned_hashing::RMIHash<Data, 10000> new_rmi(
          dataset.begin(), dataset.end(), dataset_size, true);

      EXPECT_EQ(old_rmi, new_rmi);
    }
  }
}

// ==== MonotoneRMI ====
TEST(MonotoneRMI, IsMonotone) {
  using Data = std::uint64_t;

  // generate test datasets
  std::vector<std::vector<Data>> datasets{
      {1, 2, 4, 7, 10, 1000},
      dataset::load_cached(dataset::ID::GAPPED_10, 10000)};

  for (const auto& dataset : datasets) {
    // build monotone rmi model
    const learned_hashing::MonotoneRMIHash<Data, 4> mon_rmi(
        dataset.begin(), dataset.end(), dataset.size());

    // test monotony
    size_t last_i = 0;
    for (Data k = *std::min_element(dataset.begin(), dataset.end());
         k < *std::max_element(dataset.begin(), dataset.end()); k++) {
      size_t new_i = mon_rmi(k);
      EXPECT_GE(new_i, last_i);
      last_i = new_i;
    }
  }
}
