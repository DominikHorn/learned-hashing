#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "convenience/builtins.hpp"
#include "rs/builder.h"
#include "rs/radix_spline.h"

namespace learned_hashing {
template <class Data, const size_t NumRadixBits = 18,
          const size_t MaxError = 32,
          const size_t MaxModels = std::numeric_limits<size_t>::max()>
class RadixSplineHash {
  /// output range is scaled from [0, 1] to [0, full_size) via this factor
  double out_scale_fac;

  /// internal model, a radix spline
  _rs::RadixSpline<Data> spline;

 public:
  RadixSplineHash() = default;

  template <class RandomIt>
  RadixSplineHash(const RandomIt& sample_begin, const RandomIt& sample_end,
                  const size_t full_size) {
    train(sample_begin, sample_end, full_size);
  }

  template <class RandomIt>
  void train(const RandomIt& sample_begin, const RandomIt& sample_end,
             const size_t full_size) {
    // output \in [0, sample_size] -> multiply with (full_size / sample_size)
    out_scale_fac =
        static_cast<double>(full_size - 1) /
        static_cast<double>(std::distance(sample_begin, sample_end));

    // convert input to datapoints
    const Data min = *sample_begin;
    const Data max = *(sample_end - 1);
    _rs::Builder<Data> rsb(min, max, NumRadixBits, MaxError);
    for (auto it = sample_begin; it < sample_end; it++) rsb.AddKey(*it);

    // actually build radix spline
    spline = rsb.Finalize();

    // check that we're within accepted bounds of MaxModels
    if (spline.spline_points_.size() > MaxModels)
      throw std::runtime_error("RS " + name() +
                               " had more models than allowed: " +
                               std::to_string(spline.spline_points_.size()) +
                               " > " + std::to_string(MaxModels));
  }

  template <class Result = size_t>
  forceinline Result operator()(const Data& key) const {
    return static_cast<Result>(spline.GetEstimatedPosition(key) *
                               out_scale_fac);
  }

  size_t model_count() const { return spline.spline_points_.size(); }

  size_t byte_size() const {
    return sizeof(decltype(out_scale_fac)) + spline.GetSize();
  }

  static std::string name() {
    return "radix_spline_err" + std::to_string(MaxError) + "_rbits" +
           std::to_string(NumRadixBits);
  }
};
}  // namespace learned_hashing
