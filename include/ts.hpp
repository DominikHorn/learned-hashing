#pragma once

#include "convenience/builtins.hpp"

#include "ts/builder.h"
#include "ts/ts.h"

namespace learned_hashing {
template <class Data, size_t max_error = 16> class TrieSplineHash {
  /// output range is scaled from [0, sample_size) to [0, full_size) via this
  /// factor
  double _out_scale_fac;

  /// internal trie spline model, possibly trained on sample
  ts::TrieSpline<Data> _spline;

public:
  TrieSplineHash() = default;

  template <class RandomIt>
  TrieSplineHash(const RandomIt &sample_begin, const RandomIt &sample_end,
                 const size_t &full_size) {
    train(sample_begin, sample_end, full_size);
  }

  template <class RandomIt>
  void train(const RandomIt &sample_begin, const RandomIt &sample_end,
             const size_t &full_size) {
    const auto sample_size = std::distance(sample_begin, sample_end);
    // output \in [0, sample_size] -> multiply with (full_size / sample_size)
    _out_scale_fac =
        static_cast<double>(full_size - 1) / static_cast<double>(sample_size);

    // convert input to datapoints
    const Data min = *sample_begin;
    const Data max = *(sample_end - 1);
    ts::Builder<Data> tsb(min, max, max_error);
    for (auto it = sample_begin; it < sample_end; it++)
      tsb.AddKey(*it);

    // actually build radix spline
    _spline = tsb.Finalize();
  }

  forceinline size_t operator()(const Data &key) const {
    return _spline.GetEstimatedPosition(key) * _out_scale_fac;
  }

  size_t model_count() const { return _spline.spline_points_.size(); }

  size_t byte_size() const {
    return sizeof(decltype(_out_scale_fac)) + _spline.GetSize();
  }

  static std::string name() {
    return "trie_spline_err" + std::to_string(max_error);
  }
};
} // namespace learned_hashing
