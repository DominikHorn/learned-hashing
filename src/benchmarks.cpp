#include <benchmark/benchmark.h>

static void BM_LearnedHash(benchmark::State& state) {
  // TODO: implement
  for (auto _ : state) {
  }
}

BENCHMARK(BM_LearnedHash);

BENCHMARK_MAIN();
