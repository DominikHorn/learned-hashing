#include <benchmark/benchmark.h>

#include <chrono>
#include <cstdint>
#include <learned_hashing.hpp>
#include <random>

#include "./support/datasets.hpp"
#include "./support/probing_set.hpp"

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
const std::vector<std::int64_t> probe_distributions{
    static_cast<std::underlying_type_t<dataset::ProbingDistribution>>(
        dataset::ProbingDistribution::UNIFORM),
    static_cast<std::underlying_type_t<dataset::ProbingDistribution>>(
        dataset::ProbingDistribution::EXPONENTIAL)};
const std::vector<std::int64_t> sample_sizes{1, 10, 100};

template <class Hashfn>
static void BM_build_and_throughput(benchmark::State& state) {
  const auto ds_size = state.range(0);
  const auto ds_id = static_cast<dataset::ID>(state.range(1));
  const double sample_size = static_cast<double>(state.range(2)) / 100.0;

  // load dataset
  auto dataset = dataset::load_cached(ds_id, ds_size);
  if (dataset.empty()) throw std::runtime_error("benchmark dataset empty");

  // shuffle dataset to pick sample uniform randomly
  const auto shuffle_start_time = std::chrono::steady_clock::now();
  std::random_device rd_dev;
  std::default_random_engine rng(rd_dev());
  std::shuffle(dataset.begin(), dataset.end(), rng);
  const auto shuffle_end_time = std::chrono::steady_clock::now();

  const auto sample_start_time = std::chrono::steady_clock::now();
  const auto sample_n = dataset.size() * sample_size;
  std::vector<typename decltype(dataset)::value_type> sample(
      dataset.begin(), dataset.begin() + sample_n);
  const auto sample_end_time = std::chrono::steady_clock::now();

  const auto samplesort_start_time = std::chrono::steady_clock::now();
  std::sort(sample.begin(), sample.end());
  const auto samplesort_end_time = std::chrono::steady_clock::now();

  const auto build_start_time = std::chrono::steady_clock::now();
  const Hashfn hashfn(sample.begin(), sample.end(), dataset.size());
  const auto build_end_time = std::chrono::steady_clock::now();

  // probe in random order to limit caching effects
  const auto probing_dist =
      static_cast<dataset::ProbingDistribution>(state.range(3));
  const auto probing_set = dataset::generate_probing_set(dataset, probing_dist);

  // alternatively, we could hash once per outer loop iteration. However, the
  // overhead due to gbench is too high for meaningful measurements of the
  // fastest hashfns.
  size_t i = 0;
  for (auto _ : state) {
    // get next lookup element
    while (unlikely(i >= probing_set.size())) i -= probing_set.size();
    const auto key = probing_set[i++];

    // query element
    const auto pred_rank = hashfn(key);
    benchmark::DoNotOptimize(pred_rank);

    // prevent interleaved execution
    __sync_synchronize();
  }

  state.counters["shuffle_time"] =
      std::chrono::duration<double>(shuffle_end_time - shuffle_start_time)
          .count();
  state.counters["sample_time"] =
      std::chrono::duration<double>(sample_end_time - sample_start_time)
          .count();
  state.counters["samplesort_time"] =
      std::chrono::duration<double>(samplesort_end_time - samplesort_start_time)
          .count();
  state.counters["build_time"] =
      std::chrono::duration<double>(build_end_time - build_start_time).count();
  state.counters["dataset_size"] = dataset.size();
  state.counters["sample_size"] = sample_size;

  state.SetLabel(Hashfn::name() + ":" + dataset::name(ds_id));

  state.SetItemsProcessed(static_cast<size_t>(state.iterations()));
  state.SetBytesProcessed(static_cast<size_t>(state.iterations()) *
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

  // shuffle dataset to pick sample uniform randomly
  std::random_device rd_dev;
  std::default_random_engine rng(rd_dev());
  std::shuffle(dataset.begin(), dataset.end(), rng);

  const auto sample_n = dataset.size() * sample_size;
  std::vector<typename decltype(dataset)::value_type> sample(
      dataset.begin(), dataset.begin() + sample_n);
  std::sort(sample.begin(), sample.end());

  const auto N = 100;
  std::array<size_t, N> buckets;
  std::fill(buckets.begin(), buckets.end(), 0);

  const Hashfn hashfn(sample.begin(), sample.end(), N);

  for (auto _ : state) {
    for (const auto& key : dataset) {
      const auto pred_rank = hashfn(key);
      const auto rank = std::min(pred_rank, buckets.size() - 1);
      buckets[rank]++;
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

#define BM(Hashfn)                                                            \
  BENCHMARK_TEMPLATE(BM_scattering, Hashfn)                                   \
      ->ArgsProduct({scattering_ds_sizes, datasets, sample_sizes})            \
      ->Iterations(1);                                                        \
  BENCHMARK_TEMPLATE(BM_build_and_throughput, Hashfn)                         \
      ->ArgsProduct(                                                          \
          {throughput_ds_sizes, datasets, sample_sizes, probe_distributions}) \
      ->Iterations(50000000)                                                  \
      ->Repetitions(3);

#define SINGLE_ARG(...) __VA_ARGS__

/// used to measure loop overhead
template <class T>
struct DoNothing {
  template <class It>
  DoNothing(const It&, const It&, const size_t) {}

  forceinline size_t operator()(const T&) const { return 0; }

  static std::string name() {
    return "DoNothing" + std::to_string(sizeof(T) * 8);
  }

  size_t byte_size() const { return 0; }
  size_t model_count() const { return 0; }
};

using Data = std::uint64_t;

BENCHMARK_TEMPLATE(BM_build_and_throughput, DoNothing<Data>)
    ->ArgsProduct({throughput_ds_sizes,
                   {static_cast<std::underlying_type_t<dataset::ID>>(
                       dataset::ID::SEQUENTIAL)},
                   {100},
                   probe_distributions})
    ->Iterations(50000000)
    ->Repetitions(3);

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
