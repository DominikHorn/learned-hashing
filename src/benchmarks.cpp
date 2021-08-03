#include <benchmark/benchmark.h>

#include <cstdint>
#include <learned_hashing.hpp>

template <class Hashfn>
static void BM_LearnedHash(benchmark::State& state) {
  using Data = std::uint64_t;
  std::vector<Data> dataset{1, 2, 3, 5, 6, 7, 8};

  Hashfn hashfn(dataset.begin(), dataset.end(), dataset.size());

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

BENCHMARK_MAIN();
