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
void histogram(const Hashfn& fn, const std::string& filepath,
               const RandomIt& begin, const RandomIt& end, double bucket_cnt) {
  // variables
  std::vector<size_t> hist(bucket_cnt, 0);

  // build historgram
  for (auto it = begin; it < end; it++) hist[fn(*it)]++;

  if (hist.empty()) return;

  std::ofstream csv_file;
  csv_file.open(filepath);
  std::cout << "writing: " << filepath << std::endl;

  csv_file << "bucket_lower,bucket_upper,bucket_value" << std::endl;
  for (size_t i = 0; i < hist.size(); i++)
    csv_file << static_cast<double>(i) / static_cast<double>(hist.size()) << ","
             << static_cast<double>(i + 1) / static_cast<double>(hist.size())
             << "," << hist[i] << std::endl;

  csv_file.close();
}

template <class Hashfn, class RandomIt>
void model(const Hashfn& fn, const std::string& filepath, const RandomIt& begin,
           const RandomIt& end) {
  std::ofstream csv_file;
  csv_file.open(filepath);
  std::cout << "writing: " << filepath << std::endl;

  const auto ds_size = std::distance(begin, end);

  csv_file << "x,y" << std::endl;
  for (auto it = begin; it < end; it += ds_size / 1000000) {
    csv_file << *it << "," << fn(*it) << std::endl;
  }

  csv_file.close();
}

template <class HashFn>
void export_all_ds(size_t dataset_size, double bucket_step = 0.000001) {
  for (const auto did :
       {dataset::ID::SEQUENTIAL, dataset::ID::GAPPED_10, dataset::ID::UNIFORM,
        dataset::ID::WIKI, dataset::ID::NORMAL, dataset::ID::OSM,
        dataset::ID::FB}) {
    const auto dataset = dataset::load_cached(did, dataset_size);
    if (dataset.empty()) continue;

    // preconditions
    std::is_sorted(dataset.begin(), dataset.end());
    const auto hist_bucket_cnt = 1.0 / bucket_step;

    // train hash function
    HashFn fn(dataset.begin(), dataset.end(), hist_bucket_cnt);

    // export histogram and fn itself
    histogram(fn,
              "stats/" + std::to_string(dataset_size / 1000000) +
                  "M/histogram/" + HashFn::name() + "_" + dataset::name(did) +
                  ".csv",
              dataset.begin(), dataset.end(), hist_bucket_cnt);
    model(fn,
          "stats/" + std::to_string(dataset_size / 1000000) + "M/models/" +
              HashFn::name() + "_" + dataset::name(did) + ".csv",
          dataset.begin(), dataset.end());
  }
}

int main() {
  using RMI = learned_hashing::RMIHash<std::uint64_t, 1000000>;
  using MonotoneRMI = learned_hashing::MonotoneRMIHash<std::uint64_t, 1000000>;

  for (auto dataset_size : {10000000, 100000000}) {
    export_all_ds<RMI>(dataset_size);
    export_all_ds<MonotoneRMI>(dataset_size);
  }

  return 0;
}
