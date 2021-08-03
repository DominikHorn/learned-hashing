# Learned Hashing

A header only cmake/c++ library that exposes various state-of-the-art
learned hash functions, i.e., learned models repurposed to fill the role
of a classical hash function.

Learned hash functions can for example aim to utilize knowledge about the
underlying data distributions to minimize collisions. One common technique
for achieving this is to learn the CDF of the dataset and use CDF(x) as the
hash function. Since CDF functions are monotone, such a hash function is also
automatically monotone:

k1 <= k2 -->  h(k1) <= h(k2)
