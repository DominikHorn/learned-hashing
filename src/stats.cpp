#include <algorithm>
#include <cstdint>
#include <fstream>
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
                              double bucket_step) {
  // preconditions
  std::is_sorted(begin, end);

  // variables
  const auto dataset_size = std::distance(begin, end);
  const auto bucket_cnt = 1.0 / bucket_step;
  std::vector<size_t> buckets(bucket_cnt, 0);

  // train hash function
  Hashfn fn(begin, end, bucket_cnt);

  // build historgram
  for (auto it = begin; it < end; it++) buckets[fn(*it)]++;

  return buckets;
}

template <class T>
void print_hist(const std::vector<size_t>& hist, double bucket_step,
                dataset::ID did, const std::vector<T>& dataset) {
  const auto max = std::max_element(hist.begin(), hist.end());
  std::cout << "Stats for " << std::fixed << std::setprecision(2)
            << dataset::name(did) << " (" << dataset.size()
            << " elems):" << std::endl;
  for (size_t i = 0; i < hist.size(); i++) {
    const auto bucket = hist[i];
    std::cout << "[" << i * bucket_step << ", " << (i + 1) * bucket_step << ") "
              << bucket << " elems: ";
    for (size_t i = 0; i < (100 * bucket / *max); i++) {
      std::cout << "*";
    }
    std::cout << std::endl;
  }
}

void hist_to_csv(const std::string& filepath, const std::vector<size_t>& hist,
                 double bucket_step) {
  if (hist.empty()) return;

  std::ofstream csv_file;
  csv_file.open(filepath);

  csv_file << "bucket_lower,bucket_upper,bucket_value" << std::endl;
  for (size_t i = 0; i < hist.size(); i++)
    csv_file << i * bucket_step << "," << (i + 1) * bucket_step << ","
             << hist[i] << std::endl;

  csv_file.close();
}

template <class HashFn>
void hist_all_ds(size_t dataset_size = 10000000, double bucket_step = 0.001) {
  for (const auto did :
       {dataset::ID::SEQUENTIAL, dataset::ID::GAPPED_10, dataset::ID::UNIFORM,
        dataset::ID::WIKI, dataset::ID::NORMAL, dataset::ID::OSM,
        dataset::ID::FB}) {
    const auto dataset = dataset::load_cached(did, dataset_size);
    if (dataset.empty()) continue;

    const auto hist =
        histogram<HashFn>(dataset.begin(), dataset.end(), bucket_step);

    hist_to_csv(HashFn::name() + "_" + dataset::name(did) + ".csv", hist,
                bucket_step);

    // std::cout << "===== " << HashFn::name() << " =====" << std::endl;
    // print_hist(hist, bucket_step, did, dataset);
  }
}

int main() {
  using RMI = learned_hashing::RMIHash<std::uint64_t, 1000000>;
  using MonotoneRMI = learned_hashing::MonotoneRMIHash<std::uint64_t, 1000000>;

  hist_all_ds<RMI>();
  hist_all_ds<MonotoneRMI>();

  return 0;
}
