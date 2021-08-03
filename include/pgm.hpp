#pragma once

#include <pgm/pgm_index.hpp>
#include <stdexcept>
#include <string>

#include "convenience/builtins.hpp"

namespace learned_hashing {

template <typename T, size_t Epsilon, size_t EpsilonRecursive,
          const size_t MaxModels = std::numeric_limits<size_t>::max(),
          typename Floating = float>
struct PGMHash : public pgm::PGMIndex<T, Epsilon, EpsilonRecursive, Floating> {
 private:
  using Parent = pgm::PGMIndex<T, Epsilon, EpsilonRecursive, Floating>;

  const T first_key;
  const size_t sample_size;
  const size_t N;

 public:
  /**
   * Constructs based on the sorted keys in the range [first, last). Note that
   * contrary to PGMIndex, a sample of the keys suffices.
   *
   * @param sample_begin, sample_end the range containing the sorted (!) keys to
   * be indexed
   * @param full_size the output range of the hash function [0, full_size)
   */
  template <typename RandomIt>
  PGMHash(const RandomIt& sample_begin, const RandomIt& sample_end,
          const size_t full_size)
      : Parent(sample_begin, sample_end),
        first_key(*sample_begin),
        sample_size(std::distance(sample_begin, sample_end)),
        N(full_size) {
    if (this->segments.size() > MaxModels) {
      throw std::runtime_error("PGM " + name() +
                               " had more models than allowed: " +
                               std::to_string(this->segments.size()) + " > " +
                               std::to_string(MaxModels));
    }
  }

  size_t model_count() { return this->segments.size(); }

  /**
   * Human readable name useful, e.g., to log measured results
   * @return
   */
  static std::string name() {
    return "pgm_hash_eps" + std::to_string(Epsilon) + "_epsrec" +
           std::to_string(EpsilonRecursive);
  }

  /**
   * Computes a hash value aiming to be within [0, N] based
   * on the PGMIndex::search algorithm. Note that additional
   * reduction might be necessary to guarantee [0, N] bounds.
   *
   * Contrary to PGMIndex::search, the precision available through
   * segment->slope is not immediately thrown away and instead carried
   * into the scaling (from input keyset size to [0, N])
   * computation, which should result in significantly more unique
   * hash/pos values compared to standard PGMIndex::search() when
   * the index is only trained on a sample.
   *
   * @tparam Result
   * @tparam Precision
   * @param key
   * @param N
   * @return
   */
  template <typename Result = size_t, typename Precision = double>
  forceinline Result operator()(const T& key) const {
    // Otherwise pgm will EXC_BAD_ACCESS
    if (unlikely(key == std::numeric_limits<T>::max())) {
      return N;
    }

    auto k = std::max(first_key, key);
    auto it = this->segment_for_key(k);

    // compute estimated pos (contrary to standard PGM, don't just throw slope
    // precision away)
    const auto first_key_in_segment = it->key;
    auto segment_pos =
        static_cast<Precision>(it->slope * (k - first_key_in_segment)) +
        it->intercept;
    Precision relative_pos = segment_pos / static_cast<double>(sample_size);
    auto global_pos =
        static_cast<Result>(static_cast<Precision>(N) * relative_pos);

    // TODO: standard pgm algorithm limits returned segment pos to at max
    // intercept of next
    //    slope segment. Maybe we should do something similar?
    //         auto pos = std::min<size_t>((*it)(k), std::next(it)->intercept);

    return global_pos;
  }
};
}  // namespace learned_hashing
