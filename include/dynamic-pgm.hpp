#pragma once

#include <memory>
#include <pgm/pgm_index.hpp>
#include <pgm/pgm_index_dynamic.hpp>
#include <stdexcept>
#include <string>

#include "convenience/builtins.hpp"

namespace learned_hashing {

template <typename T, size_t Epsilon, size_t EpsilonRecursive = Epsilon,
          typename Floating = float>
struct DynamicPGMHash {
 private:
  using PGM_T = pgm::DynamicPGMIndex<
      T, size_t, pgm::PGMIndex<T, Epsilon, EpsilonRecursive, Floating>>;
  std::unique_ptr<PGM_T> pgm_ptr_;

  T first_key_;
  double scale_fac_;

 public:
  /**
   * Constructor that produces null PGMHash.
   * Use `fit()` to initialize.
   */

  DynamicPGMHash() noexcept = default;

  /**
   * Constructs based on the sorted keys in the range [first, last). Note that
   * contrary to PGMIndex, a sample of the keys suffices.
   *
   * @param sample_begin, sample_end the range containing the sorted (!) keys to
   * be indexed
   * @param full_size the output range of the hash function [0, full_size)
   */
  template <typename RandomIt>
  DynamicPGMHash(const RandomIt &sample_begin, const RandomIt &sample_end,
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
    const auto sample_size = std::distance(sample_begin, sample_end);

    first_key_ = *sample_begin;
    scale_fac_ =
        static_cast<double>(full_size) / static_cast<double>(sample_size);

    std::vector<std::pair<T, size_t>> data;
    data.resize(sample_size);
    size_t i = 0;
    for (auto it = sample_begin; it < sample_end; it++)
      data.emplace_back(*it, i++);

    pgm_ptr_ = std::make_unique<PGM_T>(data.begin(), data.end());
  }

  /**
   * Amount of models in PGM
   */
  size_t model_count() const { return pgm_ptr_->segments.size(); }

  /**
   * Size of PGM model in bytes
   */
  size_t byte_size() const { return pgm_ptr_->size_in_bytes(); }

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
    const auto it = pgm_ptr_->lower_bound(key);

    return std::distance(pgm_ptr_->begin(), it);
  }
};
}  // namespace learned_hashing
