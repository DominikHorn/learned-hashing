#pragma once

#include <gtest/gtest.h>

#include <cstdint>
#include <learned_hashing.hpp>
#include <tuple>
#include <vector>

template <class Data, size_t I = 0, typename... Tp>
void iter_rmis(const std::tuple<Tp...>& t, const std::vector<Data>& dataset,
               const size_t dataset_size) {
  if constexpr (I + 1 != sizeof...(Tp))
    iter_rmis<Data, I + 1>(t, dataset, dataset_size);
}

// on sequential data, there mustn't be any collisions!
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
