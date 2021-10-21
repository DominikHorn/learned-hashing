#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include "convenience/builtins.hpp"

namespace learned_hashing {
template <class X, class Y>
struct DatapointImpl {
  X x;
  Y y;

  DatapointImpl(const X x, const Y y) : x(x), y(y) {}
};

template <class Key, class Precision>
struct LinearImpl {
 protected:
  Precision slope = 0, intercept = 0;

 private:
  using Datapoint = DatapointImpl<Key, Precision>;

  static forceinline Precision compute_slope(const Datapoint& min,
                                             const Datapoint& max) {
    if (min.x == max.x) return 0;

    // slope = delta(y)/delta(x)
    return ((max.y - min.y) / (max.x - min.x));
  }

  static forceinline Precision compute_intercept(const Datapoint& min,
                                                 const Datapoint& max) {
    // f(min.x) = min.y <=> slope * min.x + intercept = min.y <=> intercept =
    // min.y - slope * min.x
    return (min.y - compute_slope(min, max) * min.x);
  }

  static forceinline Precision compute_slope(const Key& minX,
                                             const Precision& minY,
                                             const Key& maxX,
                                             const Precision& maxY) {
    if (minX == maxX) return 0;

    // slope = delta(y)/delta(x)
    return ((maxY - minY) / (maxX - minX));
  }

  static forceinline Precision compute_intercept(const Key& minX,
                                                 const Precision& minY,
                                                 const Key& maxX,
                                                 const Precision& maxY) {
    // f(min.x) = min.y <=> slope * min.x + intercept = min.y <=> intercept =
    // min.y - slope * min.x
    return (minY - compute_slope(minX, minY, maxX, maxY) * minX);
  }

  explicit LinearImpl(const Key& minX, const Precision& minY, const Key& maxX,
                      const Precision& maxY)
      : slope(compute_slope(minX, minY, maxX, maxY)),
        intercept(compute_intercept(minX, minY, maxX, maxY)) {}

 public:
  explicit LinearImpl(Precision slope = 0, Precision intercept = 0)
      : slope(slope), intercept(intercept) {}

  /**
   * Performs trivial linear regression on the datapoints (i.e., computing
   * max->min spline)
   *
   * @param datapoints *sorted* datapoints to 'train' on
   * @param output_range outputs will be in range [0, output_range]
   */
  explicit LinearImpl(const std::vector<Datapoint>& datapoints)
      : slope(compute_slope(datapoints.front(), datapoints.back())),
        intercept(compute_intercept(datapoints.front(), datapoints.back())) {
    assert(slope != NAN);
    assert(intercept != NAN);
  }

  /**
   * Performs trivial linear regression on the datapoints (i.e., computing
   * max->min spline).
   *
   * @param keys sorted key array
   * @param begin first key contained in the training bucket
   * @param end last key contained in the training bucket
   */
  template <class It>
  LinearImpl(const It& dataset_begin, const It& dataset_end, size_t begin,
             size_t end)
      : LinearImpl(
            *(dataset_begin + begin),
            static_cast<Precision>(begin) /
                static_cast<Precision>(
                    std::distance(dataset_begin, dataset_end)),
            *(dataset_begin + end),
            static_cast<Precision>(end) / static_cast<Precision>(std::distance(
                                              dataset_begin, dataset_end))) {}

  /**
   * Extrapolates an index for the given key to the range [0, max_value]
   *
   * @param k key value to extrapolate for
   * @param max_value output indices are \in [0, max_value]. Defaults to
   * std::numeric_limits<Precision>::max()
   */
  forceinline size_t
  operator()(const Key& k, const Precision& max_value =
                               std::numeric_limits<Precision>::max()) const {
    // (slope * k + intercept) \in [0, 1] by construction
    const size_t pred = max_value * (slope * k + intercept) + 0.5;
    assert(pred >= 0);
    assert(pred <= max_value);
    return pred;
  }

  /**
   * Two LinearImpl are equal if their slope & intercept match *exactly*
   */
  bool operator==(const LinearImpl<Key, Precision> other) const {
    return slope == other.slope && intercept == other.intercept;
  }
};

template <class Key, size_t SecondLevelModelCount,
          bool FasterConstruction = false, class Precision = double,
          class RootModel = LinearImpl<Key, Precision>,
          class SecondLevelModel = LinearImpl<Key, Precision>>
class RMIHash {
  using Datapoint = DatapointImpl<Key, Precision>;

  /// Root model
  RootModel root_model;

  /// Second level models
  std::vector<SecondLevelModel> second_level_models;

  /// output range is scaled from [0, 1] to [0, full_size)
  size_t full_size = 0;

 public:
  /**
   * Constructs an empty, untrained RMI. to train, manually
   * train by invoking the train() function
   */
  RMIHash() = default;

  /**
   * Builds rmi on an already sorted (!) sample
   * @tparam RandomIt
   * @param sample_begin
   * @param sample_end
   * @param full_size operator() will extrapolate to [0, full_size)
   */
  template <class RandomIt>
  RMIHash(const RandomIt& sample_begin, const RandomIt& sample_end,
          const size_t full_size) {
    train(sample_begin, sample_end, full_size);
  }

  /**
   * trains rmi on an already sorted sample
   *
   * @tparam RandomIt
   * @param sample_begin
   * @param sample_end
   * @param full_size operator() will extrapolate to [0, full_size)
   */
  template <class RandomIt>
  void train(const RandomIt& sample_begin, const RandomIt& sample_end,
             const size_t full_size) {
    const size_t sample_size = std::distance(sample_begin, sample_end);
    if (sample_size == 0) return;

    root_model =
        decltype(root_model)(sample_begin, sample_end, 0, sample_size - 1);
    second_level_models = decltype(second_level_models)(SecondLevelModelCount);
    this->full_size = full_size;
    if (SecondLevelModelCount == 0) return;

    if (FasterConstruction) {
      size_t previous_end = 0, finished_end = 0, last_index = 0;
      for (auto it = sample_begin; it < sample_end; it++) {
        // Predict second level model using root model and put
        // sample datapoint into corresponding training bucket
        const auto key = *it;
        const auto current_second_level_index =
            root_model(key, SecondLevelModelCount - 1);
        assert(current_second_level_index >= 0);
        assert(current_second_level_index < SecondLevelModelCount);

        // current bucket end
        size_t current_end = std::distance(sample_begin, it);

        // we finished a bucket (sample sorted!)
        if (last_index < current_second_level_index) {
          // 'train' last model
          second_level_models[last_index] = SecondLevelModel(
              sample_begin, sample_end, finished_end, previous_end);

          // since all models are initialized to (0,0), we don't need to
          // explicitely 'train' skipped models between last index and current
          // index and can simple set last_index = current_second_level_index
          last_index = current_second_level_index;
          finished_end = previous_end;
        }

        previous_end = current_end;
      }

      // train last model (otherwise this would never happen
      second_level_models[second_level_models.size() - 1] = SecondLevelModel(
          sample_begin, sample_end, finished_end, previous_end);
    } else {
      // Assign each sample point into a training bucket according to root model
      std::vector<std::vector<Datapoint>> training_buckets(
          SecondLevelModelCount);

      for (auto it = sample_begin; it < sample_end; it++) {
        const auto i = std::distance(sample_begin, it);

        // Predict second level model using root model and put
        // sample datapoint into corresponding training bucket
        const auto key = *it;
        const auto second_level_index =
            root_model(key, SecondLevelModelCount - 1);
        auto& bucket = training_buckets[second_level_index];

        // The following works because the previous training bucket has to be
        // completed, because the sample is sorted: Each training bucket's min
        // is the previous training bucket's max (except for first bucket)
        if (bucket.empty() && second_level_index > 0 &&
            !training_buckets[second_level_index - 1].empty())
          bucket.push_back(training_buckets[second_level_index - 1].back());

        // Add datapoint at the end of the bucket
        bucket.push_back(Datapoint(
            key,
            static_cast<Precision>(i) / static_cast<Precision>(sample_size)));
      }

      // Edge case: First model does not have enough training data -> add
      // artificial datapoints
      while (training_buckets[0].size() < 2)
        training_buckets[0].insert(training_buckets[0].begin(),
                                   Datapoint(0, 0));

      // Train each second level model on its respective bucket
      for (size_t model_idx = 0; model_idx < SecondLevelModelCount;
           model_idx++) {
        auto& training_bucket = training_buckets[model_idx];

        // Propagate datapoints from previous training bucket if necessary
        while (training_bucket.size() < 2) {
          assert(model_idx - 1 >= 0);
          assert(!training_buckets[model_idx - 1].empty());
          training_bucket.insert(training_bucket.begin(),
                                 training_buckets[model_idx - 1].back());
        }
        assert(training_bucket.size() >= 2);

        // Train model on training bucket & add it
        second_level_models[model_idx] = SecondLevelModel(training_bucket);
      }
    }
  }

  static std::string name() {
    return "rmi_hash_" + std::to_string(SecondLevelModelCount);
  }

  size_t byte_size() const {
    return sizeof(decltype(this)) +
           sizeof(SecondLevelModel) * SecondLevelModelCount;
  }

  size_t model_count() { return 1 + SecondLevelModelCount; }

  /**
   * Compute hash value for key
   *
   * @tparam Result result data type. Defaults to size_t
   * @param key
   */
  template <class Result = size_t>
  forceinline Result operator()(const Key& key) const {
    if (SecondLevelModelCount == 0) return root_model(key, full_size);

    const auto second_level_index = root_model(key, SecondLevelModelCount - 1);
    return second_level_models[second_level_index](key, full_size);
  }

  bool operator==(const RMIHash<Key, SecondLevelModelCount> other) const {
    if (other.root_model != root_model) return false;
    for (size_t i = 0; i < SecondLevelModelCount; i++)
      if (other.second_level_models[i] != second_level_models[i]) return false;

    return true;
  }
};
}  // namespace learned_hashing

