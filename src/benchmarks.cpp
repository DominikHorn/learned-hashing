#include <benchmark/benchmark.h>

#include <chrono>
#include <cstdint>
#include <learned_hashing.hpp>
#include <random>

#include "./support/datasets.hpp"

const std::vector<std::int64_t> throughput_ds_sizes{1'000'000, 10'000'000,
                                                    200'000'000};
const std::vector<std::int64_t> scattering_ds_sizes{10'000'000};
const std::vector<std::int64_t> datasets{
    static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::SEQUENTIAL),
    static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::GAPPED_10),
    static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::UNIFORM),
    static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::NORMAL),
    static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::BOOKS),
    static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::FB),
    static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::OSM),
    static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::WIKI)};
const std::vector<std::int64_t> sample_sizes{1, 10, 50, 100};

template <class Hashfn>
static void BM_build_and_throughput(benchmark::State& state) {
  const auto ds_size = state.range(0);
  const auto ds_id = static_cast<dataset::ID>(state.range(1));
  const double sample_size = static_cast<double>(state.range(2)) / 100.0;

  // load dataset
  auto dataset = dataset::load_cached(ds_id, ds_size);
  if (dataset.empty()) throw std::runtime_error("benchmark dataset empty");

  // shuffle dataset
  std::random_device rd_dev;
  std::default_random_engine rng(rd_dev());
  std::shuffle(dataset.begin(), dataset.end(), rng);

  const auto sample_n = dataset.size() * sample_size;
  const std::vector<typename decltype(dataset)::value_type> sample(
      dataset.begin(), dataset.begin() + sample_n);

  const auto build_start_time = std::chrono::steady_clock::now();
  const Hashfn hashfn(sample.begin(), sample.end(), dataset.size());
  const auto build_end_time = std::chrono::steady_clock::now();

  // alternatively, we could hash once per outer loop iteration. However, the
  // overhead due to gbench is too high for meaningful measurements of the
  // fastest hashfns.
  for (auto _ : state) {
    for (const auto& key : dataset) {
      const auto pred_rank = hashfn(key);
      benchmark::DoNotOptimize(pred_rank);
      __sync_synchronize();
    }
  }

  state.counters["build_time"] =
      std::chrono::duration<double>(build_end_time - build_start_time).count();
  state.counters["dataset_size"] = dataset.size();
  state.counters["sample_size"] = sample_size;

  state.SetLabel(Hashfn::name() + ":" + dataset::name(ds_id));

  state.SetItemsProcessed(dataset.size() *
                          static_cast<size_t>(state.iterations()));
  state.SetBytesProcessed(dataset.size() *
                          static_cast<size_t>(state.iterations()) *
                          sizeof(typename decltype(dataset)::value_type));
}

template <class Hashfn>
static void BM_scattering(benchmark::State& state) {
  const auto ds_size = state.range(0);
  const auto ds_id = static_cast<dataset::ID>(state.range(1));
  const double sample_size = static_cast<double>(state.range(2)) / 100.0;

  // load dataset
  auto dataset = dataset::load_cached(ds_id, ds_size);
  if (dataset.empty()) throw std::runtime_error("benchmark dataset empty");

  // shuffle dataset
  std::random_device rd_dev;
  std::default_random_engine rng(rd_dev());
  std::shuffle(dataset.begin(), dataset.end(), rng);

  const auto sample_n = dataset.size() * sample_size;
  const std::vector<typename decltype(dataset)::value_type> sample(
      dataset.begin(), dataset.begin() + sample_n);

  const auto N = 100;
  std::array<size_t, N> buckets;
  std::fill(buckets.begin(), buckets.end(), 0);

  const Hashfn hashfn(sample.begin(), sample.end(), N);

  for (auto _ : state) {
    for (const auto& key : dataset) {
      const auto pred_rank = hashfn(key);
      buckets[pred_rank]++;
    }
  }

  for (size_t i = 0; i < N; i++)
    state.counters["bucket_" + std::to_string(i)] = buckets[i];

  state.counters["dataset_size"] = dataset.size();
  state.counters["sample_size"] = sample_size;

  state.SetLabel(Hashfn::name() + ":" + dataset::name(ds_id));

  state.SetItemsProcessed(dataset.size() *
                          static_cast<size_t>(state.iterations()));
  state.SetBytesProcessed(dataset.size() *
                          static_cast<size_t>(state.iterations()) *
                          sizeof(typename decltype(dataset)::value_type));
}

#define BM(Hashfn)                                                 \
  BENCHMARK_TEMPLATE(BM_build_and_throughput, Hashfn)              \
      ->ArgsProduct({throughput_ds_sizes, datasets, sample_sizes}) \
      ->Iterations(5)                                              \
      ->Repetitions(5);                                            \
  BENCHMARK_TEMPLATE(BM_scattering, Hashfn)                        \
      ->ArgsProduct({scattering_ds_sizes, datasets, sample_sizes}) \
      ->Iterations(1);

#define SINGLE_ARG(...) __VA_ARGS__

BM(SINGLE_ARG(learned_hashing::RMIHash<std::uint64_t, 1'000'000>));
BM(SINGLE_ARG(learned_hashing::RMIHash<std::uint64_t, 10'000>));
BM(SINGLE_ARG(learned_hashing::RMIHash<std::uint64_t, 100>));

BM(SINGLE_ARG(learned_hashing::PGMHash<std::uint64_t, 4>));
BM(SINGLE_ARG(learned_hashing::PGMHash<std::uint64_t, 16>));
BM(SINGLE_ARG(learned_hashing::PGMHash<std::uint64_t, 128>));

BM(SINGLE_ARG(learned_hashing::CHTHash<std::uint64_t, 4>));
BM(SINGLE_ARG(learned_hashing::CHTHash<std::uint64_t, 16>));
BM(SINGLE_ARG(learned_hashing::CHTHash<std::uint64_t, 128>));

BM(SINGLE_ARG(learned_hashing::RadixSplineHash<std::uint64_t, 18, 4>));
BM(SINGLE_ARG(learned_hashing::RadixSplineHash<std::uint64_t, 18, 16>));
BM(SINGLE_ARG(learned_hashing::RadixSplineHash<std::uint64_t, 18, 128>));

BM(SINGLE_ARG(learned_hashing::TrieSplineHash<std::uint64_t, 4>));
BM(SINGLE_ARG(learned_hashing::TrieSplineHash<std::uint64_t, 16>));
BM(SINGLE_ARG(learned_hashing::TrieSplineHash<std::uint64_t, 128>));

BENCHMARK_MAIN();
