#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <ios>
#include <iostream>
#include <iterator>
#include <learned_hashing.hpp>
#include <string>
#include <vector>

#include "include/rmi.hpp"
#include "support/datasets.hpp"

/**
 * Builds a distribution histogram over the entire dataset given a certain
 * hashfunction, dataset and bucket size
 *
 * bucket_size \in [0, 1]
 */
template <class Hashfn, class RandomIt>
std::vector<size_t> histogram(const RandomIt& begin, const RandomIt& end,
                              double bucket_size) {
  // preconditions
  std::is_sorted(begin, end);

  // variables
  const auto dataset_size = std::distance(begin, end);
  const auto bucket_cnt = 1.0 / bucket_size;
  std::vector<size_t> buckets(bucket_cnt, 0);

  // train hash function
  Hashfn fn(begin, end, bucket_cnt);

  // build historgram
  for (auto it = begin; it < end; it++) buckets[fn(*it)]++;

  return buckets;
}

int main() {
  using RMI = learned_hashing::RMIHash<std::uint64_t, 1000000>;

  const auto dataset_size = 10000000;
  const auto bucket_step = 0.02;

  for (const auto did :
       {dataset::ID::SEQUENTIAL, dataset::ID::GAPPED_10, dataset::ID::UNIFORM,
        dataset::ID::WIKI, dataset::ID::NORMAL, dataset::ID::OSM,
        dataset::ID::FB}) {
    const auto dataset = dataset::load_cached(did, dataset_size);
    if (dataset.empty()) continue;

    const auto hist =
        histogram<RMI>(dataset.begin(), dataset.end(), bucket_step);

    // temporary export code
    const auto max = std::max_element(hist.begin(), hist.end());
    std::cout << "Stats for " << std::fixed << std::setprecision(2)
              << dataset::name(did) << " (" << dataset.size()
              << " elems):" << std::endl;
    for (size_t i = 0; i < hist.size(); i++) {
      const auto bucket = hist[i];
      std::cout << "[" << i * bucket_step << ", " << (i + 1) * bucket_step
                << "): ";
      for (size_t i = 0; i < (100 * bucket / *max); i++) {
        std::cout << "*";
      }
      std::cout << std::endl;
    }
  }

  return 0;
}
