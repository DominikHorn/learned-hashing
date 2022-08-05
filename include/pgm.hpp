#pragma once

#include <pgm/pgm_index.hpp>
#include <stdexcept>
#include <string>

#include "convenience/builtins.hpp"

namespace learned_hashing {

template <typename T, size_t Epsilon, size_t EpsilonRecursive = Epsilon,
          const size_t MaxModels = std::numeric_limits<size_t>::max(),
          typename Floating = float>
struct PGMHash {
 private:
  pgm::PGMIndex<T, Epsilon, EpsilonRecursive, Floating> pgm_;

  T first_key_;
  double scale_fac_;

 public:
  /**
   * Constructor that produces null PGMHash.
   * Use `fit()` to initialize.
   */

  PGMHash() noexcept = default;

  /**
   * Constructs based on the sorted keys in the range [first, last). Note that
   * contrary to PGMIndex, a sample of the keys suffices.
   *
   * @param sample_begin, sample_end the range containing the sorted (!) keys to
   * be indexed
   * @param full_size the output range of the hash function [0, full_size)
   */
  template <typename RandomIt>
  PGMHash(const RandomIt &sample_begin, const RandomIt &sample_end,
          const size_t full_size) {
    train(sample_begin, sample_end, full_size);
  }

  /**
   * Fits this PGMHash instance to a certain data distribution based on a
  sample.
   *
   * @param sample_begin iterator to first element of the sample
   * @param sample_end past the end iterator for sample
   * @param full_size actual full dataset size
   */
  template <class RandomIt>
  void train(const RandomIt &sample_begin, const RandomIt &sample_end,
             const size_t full_size) {
    first_key_ = *sample_begin;

    const auto sample_size_ = std::distance(sample_begin, sample_end);
    scale_fac_ =
        static_cast<double>(full_size) / static_cast<double>(sample_size_);

    pgm_ = decltype(pgm_)(sample_begin, sample_end);
    if (pgm_.segments.size() > MaxModels) {
      throw std::runtime_error("PGM " + name() +
                               " had more models than allowed: " +
                               std::to_string(pgm_.segments.size()) + " > " +
                               std::to_string(MaxModels));
    }
  }

  /**
   * Amount of models in PGM
   */
  size_t model_count() const { return pgm_.segments.size(); }

  /**
   * Size of PGM model in bytes
   */
  size_t byte_size() const { return pgm_.size_in_bytes(); }

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
  forceinline Result operator()(const T &key) const {
    // Otherwise pgm will EXC_BAD_ACCESS
    if (unlikely(key == std::numeric_limits<T>::max())) {
      return std::numeric_limits<T>::max();
    }

    return static_cast<Result>(scale_fac_ * pgm_.get_pos(key));
  }
};
}  // namespace learned_hashing
