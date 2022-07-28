#pragma once

#include <cstdint>
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
      T, std::uint8_t, pgm::PGMIndex<T, Epsilon, EpsilonRecursive, Floating>>;
  std::unique_ptr<PGM_T> pgm_ptr_;

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
   * @param keys_begin, keys_end the range containing the sorted (!) keys to
   * be indexed
   */
  template <typename RandomIt>
  DynamicPGMHash(const RandomIt &keys_begin, const RandomIt &keys_end) {
    train(keys_begin, keys_end);
  }

  /**
   * Fits this PGMHash instance to a certain data distribution based on a
  sample.
   *
   * @param keys_begin iterator to first key
   * @param keys_end past the end iterator for keys
   */
  template <class RandomIt>
  void train(const RandomIt &keys_begin, const RandomIt &keys_end) {
    const auto N = std::distance(keys_begin, keys_end);

    std::vector<std::pair<T, std::uint8_t>> data;
    data.resize(N);
    for (auto it = keys_begin; it < keys_end; it++) data.emplace_back(*it, 0);

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
    return std::distance(pgm_ptr_->begin(), pgm_ptr_->lower_bound(key));
  }

  /// inserts a new key
  void insert(const T &key) { pgm_ptr_->insert_or_assign(key, 0); }
};
}  // namespace learned_hashing
