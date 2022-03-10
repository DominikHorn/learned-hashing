#pragma once

#include "convenience/builtins.hpp"

#include "cht/builder.h"
#include "cht/cht.h"
#include "include/convenience/bounds.hpp"

namespace learned_hashing {
template <class Data, size_t max_error = 32, size_t num_bins = 64>
class CHTHash {
  /// output range is scaled from [0, sample_size) to [0, full_size) via this
  /// factor
  double _out_scale_fac;

  /// underlying model
  cht::CompactHistTree<Data> _cht;

public:
  CHTHash() noexcept = default;

  template <class RandomIt>
  CHTHash(const RandomIt &sample_begin, const RandomIt &sample_end,
          const size_t &full_size) {
    train(sample_begin, sample_end, full_size);
  }

  /// [sample_begin, sample_end) must be sorted
  template <class ForwardIt>
  void train(const ForwardIt &sample_begin, const ForwardIt &sample_end,
             const size_t &full_size) {
    const auto sample_size = std::distance(sample_begin, sample_end);
    // output \in [0, sample_size] -> multiply with (full_size / sample_size)
    _out_scale_fac =
        static_cast<double>(full_size - 1) / static_cast<double>(sample_size);

    // convert input to datapoints
    const auto min = *sample_begin;
    const auto max = *(sample_end - 1);
    cht::Builder<Data> chsb(min, max, num_bins, max_error);
    for (auto it = sample_begin; it < sample_end; it++)
      chsb.AddKey(*it);

    // actually build cht
    _cht = chsb.Finalize();
  }

  forceinline size_t operator()(const Data &key) const {
    return _cht.Lookup(key) * _out_scale_fac;
  }

  forceinline Bounds bounds(const Data &key) const {
    return _cht.GetSearchBound(key);
  }

  size_t model_size() const { return _cht.GetSize(); }

  size_t byte_size() const { return sizeof(decltype(*this)) + model_size(); }

  static std::string name() {
    return "cht_" + std::to_string(num_bins) + "_" + std::to_string(max_error);
  }
};
} // namespace learned_hashing
