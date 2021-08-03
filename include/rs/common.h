#pragma once

#include <cstddef>
#include <cstdint>

namespace learned_hashing {
namespace _rs {

// A CDF coordinate.
template <class KeyType>
struct Coord {
  KeyType x;
  double y;
};

struct SearchBound {
  size_t begin;
  size_t end;  // Exclusive.
};

}  // namespace _rs
}  // namespace learned_hashing
