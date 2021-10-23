#include <benchmark/benchmark.h>

#include <cstdint>
#include <learned_hashing.hpp>
#include <random>

template <class Hashfn>
static void BM_LearnedHash(benchmark::State& state) {
  using Data = std::uint64_t;
  std::vector<Data> dataset(100000000);
  for (Data i = 0; i < dataset.size(); i++) dataset[i] = i + 20000;

  std::random_device rd;
  std::default_random_engine rng(rd());
  std::uniform_int_distribution<size_t> dist(0, dataset.size() - 1);
  std::vector<Data> sample(dataset.size() / 100);
  for (size_t i = 0; i < sample.size(); i++) sample[i] = dataset[dist(rng)];
  std::sort(sample.begin(), sample.end());
  sample.erase(std::unique(sample.begin(), sample.end()), sample.end());
  sample.shrink_to_fit();

  Hashfn hashfn(sample.begin(), sample.end(), dataset.size());

  for (auto _ : state) {
    for (const auto& key : dataset) {
      const auto hash_value = hashfn(key);
      benchmark::DoNotOptimize(hash_value);
    }
  }

  state.SetLabel(Hashfn::name());
  state.SetItemsProcessed(dataset.size());
  state.SetBytesProcessed(dataset.size() * sizeof(Data));
}

BENCHMARK_TEMPLATE(BM_LearnedHash,
                   learned_hashing::PGMHash<std::uint64_t, 4, 1>);
BENCHMARK_TEMPLATE(BM_LearnedHash,
                   learned_hashing::RadixSplineHash<std::uint64_t>);
BENCHMARK_TEMPLATE(BM_LearnedHash,
                   learned_hashing::RMIHash<std::uint64_t, 1000000>);
BENCHMARK_TEMPLATE(BM_LearnedHash,
                   learned_hashing::MonotoneRMIHash<std::uint64_t, 1000000>);

BENCHMARK_MAIN();
