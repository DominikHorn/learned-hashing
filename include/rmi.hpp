#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include "convenience/builtins.hpp"

namespace learned_hashing {
template <class X, class Y> struct DatapointImpl {
  X x;
  Y y;

  DatapointImpl(const X x, const Y y) : x(x), y(y) {}
};

template <class Key, class Precision> struct LinearImpl {
protected:
  Precision slope = 0, intercept = 0;

private:
  using Datapoint = DatapointImpl<Key, Precision>;

  static forceinline Precision compute_slope(const Datapoint &min,
                                             const Datapoint &max) {
    if (min.x == max.x)
      return 0;

    // slope = delta(y)/delta(x)
    return ((max.y - min.y) / (max.x - min.x));
  }

  static forceinline Precision compute_intercept(const Datapoint &min,
                                                 const Datapoint &max) {
    // f(min.x) = min.y <=> slope * min.x + intercept = min.y <=> intercept =
    // min.y - slope * min.x
    return (min.y - compute_slope(min, max) * min.x);
  }

  static forceinline Precision compute_slope(const Key &minX,
                                             const Precision &minY,
                                             const Key &maxX,
                                             const Precision &maxY) {
    if (minX == maxX)
      return 0;

    // slope = delta(y)/delta(x)
    return ((maxY - minY) / (maxX - minX));
  }

  static forceinline Precision compute_intercept(const Key &minX,
                                                 const Precision &minY,
                                                 const Key &maxX,
                                                 const Precision &maxY) {
    // f(min.x) = min.y <=> slope * min.x + intercept = min.y <=> intercept =
    // min.y - slope * min.x
    return (minY - compute_slope(minX, minY, maxX, maxY) * minX);
  }

  explicit LinearImpl(const Key &minX, const Precision &minY, const Key &maxX,
                      const Precision &maxY)
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
  explicit LinearImpl(const std::vector<Datapoint> &datapoints)
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
  LinearImpl(const It &dataset_begin, const It &dataset_end, size_t begin,
             size_t end)
      : LinearImpl(*(dataset_begin + begin),
                   static_cast<Precision>(begin) /
                       static_cast<Precision>(
                           std::distance(dataset_begin, dataset_end)),
                   *(dataset_begin + end),
                   static_cast<Precision>(end) /
                       static_cast<Precision>(
                           std::distance(dataset_begin, dataset_end))) {}

  template <class It>
  LinearImpl(const It &dataset_begin, const It &dataset_end, size_t /*begin*/,
             size_t end, Key prev_max_x, Precision prev_max_y)
      : LinearImpl(
            prev_max_x, prev_max_y,
            std::max(prev_max_x, *(dataset_begin + end)),
            std::max(prev_max_y,
                     static_cast<Precision>(end) /
                         static_cast<Precision>(
                             std::distance(dataset_begin, dataset_end) - 1))) {}

  /**
   * computes y \in [0, 1] given a certain x
   */
  forceinline Precision normalized(const Key &k) const {
    const auto res = slope * k + intercept;
    if (res > 1.0)
      return 1.0;
    if (res < 0.0)
      return 0.0;
    return res;
  }

  /**
   * computes x (rounded up) given a certain y in normalized space:
   * (y \in [0, 1]).
   */
  forceinline Key normalized_inverse(const Precision y) const {
    // y = ax + b <=> x = (y-b)/a
    // +0.5 to round up (TODO(dominik): is this necessary?)
    return 0.5 + (y - intercept) / slope;
  }

  /**
   * Extrapolates an index for the given key to the range [0, max_value]
   *
   * @param k key value to extrapolate for
   * @param max_value output indices are \in [0, max_value]. Defaults to
   * std::numeric_limits<Precision>::max()
   */
  forceinline size_t
  operator()(const Key &k, const Precision &max_value =
                               std::numeric_limits<Precision>::max()) const {
    // +0.5 as a quick&dirty ceil trick
    const size_t pred = max_value * normalized(k) + 0.5;
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

  forceinline Precision get_slope() const { return slope; }
  forceinline Precision get_intercept() const { return intercept; }
};

template <class Key, size_t MaxSecondLevelModelCount,
          size_t MinAvgDatapointsPerModel = 2, class Precision = double,
          class RootModel = LinearImpl<Key, Precision>,
          class SecondLevelModel = LinearImpl<Key, Precision>>
class RMIHash {
  using Datapoint = DatapointImpl<Key, Precision>;

  /// Root model
  RootModel root_model;

  /// Second level models
  std::vector<SecondLevelModel> second_level_models;

  /// output range is scaled from [0, 1] to [0, max_output] = [0, full_size)
  size_t max_output = 0;

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
   * @param faster_construction whether or not to use the faster construction
   *    algorithm without intermediate allocations etc. Around 100x speedup
   *    while end result is the same (! tested on various datasets)
   */
  template <class RandomIt>
  RMIHash(const RandomIt &sample_begin, const RandomIt &sample_end,
          const size_t full_size, bool faster_construction = true) {
    train(sample_begin, sample_end, full_size, faster_construction);
  }

  /**
   * trains rmi on an already sorted sample
   *
   * @tparam RandomIt
   * @param sample_begin
   * @param sample_end
   * @param full_size operator() will extrapolate to [0, full_size)
   * @param faster_construction whether or not to use the faster construction
   *    algorithm without intermediate allocations etc. Around 100x speedup
   *    while end result is the same (! tested on various datasets)
   */
  template <class RandomIt>
  void train(const RandomIt &sample_begin, const RandomIt &sample_end,
             const size_t full_size, bool faster_construction = true) {
    this->max_output = full_size - 1;
    const size_t sample_size = std::distance(sample_begin, sample_end);
    if (sample_size == 0)
      return;

    root_model =
        decltype(root_model)(sample_begin, sample_end, 0, sample_size - 1);
    if (MaxSecondLevelModelCount == 0)
      return;

    // ensure that there is at least MinAvgDatapointsPerModel datapoints per
    // model on average to not waste space/resources
    second_level_models = decltype(second_level_models)(std::min(
        MaxSecondLevelModelCount, sample_size / MinAvgDatapointsPerModel));

    if (faster_construction) {
      // convenience function for training (code deduplication)
      size_t previous_end = 0, finished_end = 0, last_index = 0;
      const auto train_until = [&](const size_t i) {
        while (last_index < i) {
          second_level_models[last_index++] = SecondLevelModel(
              sample_begin, sample_end, finished_end, previous_end);
          finished_end = previous_end;
        }
      };

      for (auto it = sample_begin; it < sample_end; it++) {
        // Predict second level model using root model and put
        // sample datapoint into corresponding training bucket
        const auto key = *it;
        const auto current_second_level_index =
            root_model(key, second_level_models.size() - 1);
        assert(current_second_level_index >= 0);
        assert(current_second_level_index < second_level_models.size());

        // bucket is finished, train all affected models
        if (last_index < current_second_level_index)
          train_until(current_second_level_index);

        // last consumed datapoint
        previous_end = std::distance(sample_begin, it);
      }

      // train all remaining models
      train_until(second_level_models.size());
    } else {
      // Assign each sample point into a training bucket according to root model
      std::vector<std::vector<Datapoint>> training_buckets(
          second_level_models.size());

      for (auto it = sample_begin; it < sample_end; it++) {
        const auto i = std::distance(sample_begin, it);

        // Predict second level model using root model and put
        // sample datapoint into corresponding training bucket
        const auto key = *it;
        const auto second_level_index =
            root_model(key, second_level_models.size() - 1);
        auto &bucket = training_buckets[second_level_index];

        // The following works because the previous training bucket has to be
        // completed, because the sample is sorted: Each training bucket's min
        // is the previous training bucket's max (except for first bucket)
        if (bucket.empty() && second_level_index > 0) {
          size_t j = second_level_index - 1;
          while (j < second_level_index && j >= 0 &&
                 training_buckets[j].empty())
            j--;
          assert(!training_buckets[j].empty());
          bucket.push_back(training_buckets[j].back());
        }

        // Add datapoint at the end of the bucket
        bucket.push_back(
            Datapoint(key, static_cast<Precision>(i) /
                               static_cast<Precision>(sample_size)));
      }

      // Edge case: First model does not have enough training data -> add
      // artificial datapoints
      assert(training_buckets[0].size() >= 1);
      while (training_buckets[0].size() < 2)
        training_buckets[0].insert(training_buckets[0].begin(),
                                   Datapoint(0, 0));

      // Train each second level model on its respective bucket
      for (size_t model_idx = 0; model_idx < second_level_models.size();
           model_idx++) {
        auto &training_bucket = training_buckets[model_idx];

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
    return "rmi_hash_" + std::to_string(MaxSecondLevelModelCount);
  }

  size_t byte_size() const {
    return sizeof(decltype(this)) +
           sizeof(SecondLevelModel) * second_level_models.size();
  }

  size_t model_count() { return 1 + second_level_models.size(); }

  /**
   * Compute hash value for key
   *
   * @tparam Result result data type. Defaults to size_t
   * @param key
   */
  template <class Result = size_t>
  forceinline Result operator()(const Key &key) const {
    if (MaxSecondLevelModelCount == 0)
      return root_model(key, max_output);

    const auto second_level_index =
        root_model(key, second_level_models.size() - 1);
    const auto result =
        second_level_models[second_level_index](key, max_output);

    assert(result <= max_output);
    return result;
  }

  bool operator==(const RMIHash &other) const {
    if (other.root_model != root_model)
      return false;
    if (other.second_level_models.size() != second_level_models.size())
      return false;

    for (size_t i = 0; i < second_level_models.size(); i++)
      if (other.second_level_models[i] != second_level_models[i])
        return false;

    return true;
  }
};

/**
 * Like RMIHash, but monotone even for non-keys due to modified
 * construction algorithm. As of writing, only implemented
 * using LinearImpl models
 */
template <class Key, size_t MaxSecondLevelModelCount,
          size_t MinAvgDatapointsPerModel = 2, class Precision = double,
          class RootModel = LinearImpl<Key, Precision>,
          class SecondLevelModel = LinearImpl<Key, Precision>>
class MonotoneRMIHash {
  using Datapoint = DatapointImpl<Key, Precision>;
  using Model = LinearImpl<Key, Precision>;

  /// Root model
  RootModel root_model;

  /// Second level models
  std::vector<SecondLevelModel> second_level_models;

  /// output range is scaled from [0, 1] to [0, max_output] = [0, full_size)
  size_t full_size = 0;

public:
  /**
   * Constructs an empty, untrained RMI. to train, manually
   * train by invoking the train() function
   */
  MonotoneRMIHash() = default;

  /**
   * Builds rmi on an already sorted (!) sample
   * @tparam RandomIt
   * @param sample_begin
   * @param sample_end
   * @param full_size operator() will extrapolate to [0, full_size)
   */
  template <class RandomIt>
  MonotoneRMIHash(const RandomIt &sample_begin, const RandomIt &sample_end,
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
  void train(const RandomIt &sample_begin, const RandomIt &sample_end,
             const size_t full_size) {
    this->full_size = full_size;
    const size_t sample_size = std::distance(sample_begin, sample_end);
    if (sample_size == 0)
      return;

    // train root model
    root_model = decltype(root_model)(sample_begin, sample_end, 0,
                                      sample_size - 1, *sample_begin, 0.0);

    assert(root_model.normalized(*sample_begin) <= 0.0001);
    assert(root_model.normalized(*(sample_end - 1)) >= 0.9999);

    // special case: single level model
    if (MaxSecondLevelModelCount == 0)
      return;

    // ensure that there is at least MinAvgDatapointsPerModel datapoints per
    // model on average to not waste space/resources
    second_level_models = decltype(second_level_models)(std::min(
        MaxSecondLevelModelCount, sample_size / MinAvgDatapointsPerModel));

    // finds (virtual) true min datapoint for training bucket/second level model
    // i such that monotony is retained even for non-keys that fit in between
    // actual keys present in the dataset
    const auto true_min_x = [&](const size_t i) {
      if (i == 0)
        return *sample_begin;
      return root_model.normalized_inverse(
          static_cast<double>(i) /
          static_cast<double>(second_level_models.size()));
    };
    const auto true_min_y = [&](const size_t i, const size_t i_min_x) {
      if (i == 0)
        return 0.0;
      const auto prev_max_y = second_level_models[i - 1].normalized(i_min_x);
      return prev_max_y;
    };

    // convenience function for training (code deduplication)
    size_t previous_end = 0, finished_end = 0, last_index = 0;
    const auto train_until = [&](const size_t i) {
      while (last_index < i) {
        const auto prev_max_x = true_min_x(last_index);
        const auto prev_max_y = true_min_y(last_index, prev_max_x);
        second_level_models[last_index++] =
            SecondLevelModel(sample_begin, sample_end, finished_end,
                             previous_end, prev_max_x, prev_max_y);
        finished_end = previous_end;
      }
    };

    // train second level models
    for (auto it = sample_begin; it < sample_end; it++) {
      // Predict second level model using root model and put
      // sample datapoint into corresponding training bucket
      const auto key = *it;
      const size_t current_second_level_index =
          root_model.normalized(key) * second_level_models.size();
      assert(current_second_level_index >= 0);
      assert(current_second_level_index <= second_level_models.size());

      // bucket is finished, train all affected models up until
      // the current model to train
      if (last_index < current_second_level_index)
        train_until(current_second_level_index);

      previous_end = std::distance(sample_begin, it);
    }

    // train remaining models
    train_until(second_level_models.size());
  }

  static std::string name() {
    return "monotone_rmi_hash_" + std::to_string(MaxSecondLevelModelCount);
  }

  size_t byte_size() const {
    return sizeof(decltype(this)) + sizeof(Model) * second_level_models.size();
  }

  size_t model_count() { return 1 + second_level_models.size(); }

  /**
   * Compute hash value for key
   *
   * @tparam Result result data type. Defaults to size_t
   * @param key
   */
  template <class Result = size_t>
  forceinline Result operator()(const Key &key) const {
    if (MaxSecondLevelModelCount == 0)
      return root_model(key, full_size);

    const size_t second_level_index =
        root_model.normalized(key) * second_level_models.size();

    if (unlikely(second_level_index >= second_level_models.size()))
      return full_size - 1;

    const size_t res =
        second_level_models[second_level_index].normalized(key) * full_size;

    return res - ((res >= full_size) & 0x1);
  }
};
} // namespace learned_hashing
