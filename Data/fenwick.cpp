template <typename T>
struct BIT {
  vector<T> fenw;
  int n, pw;

  BIT(int n_ = 0) : n(n_) {
    fenw.assign(n + 1, 0);
    pw = bit_floor(unsigned(n));
  }

  void Modify(int x, T v) {
    if (x <= 0) return;  // assert(0 <= x && x < n);
    while (x <= n) {     // x < n
      fenw[x] += v;
      x += (x & -x);  // x |= x + 1;
    }
  }

  T Query(int x) {
    // assert(0 <= x && x <= n);
    T v{};
    while (x > 0) {
      v += fenw[x];   // fenw[x - 1];
      x -= (x & -x);  // x &= x - 1;
    }
    return v;
  }

  // Returns the length of the longest prefix with sum <= c
  int MaxPrefix(T c) {
    T v{};
    int at = 0;
    for (int i = 20; i >= 0; i--) {
      if (at + bit(i) <= n && v + fenw[at + bit(i)] <= c) {
        v += fenw[at + bit(i)];
        at += bit(i);
      }
    }
    /**
     * for (int len = pw; len > 0; len >>= 1) {
     *   if (at + len <= n) {
     *     auto nv = v;
     *     nv += fenw[at + len - 1];
     *     if (!(c < nv)) {
     *       v = nv;
     *       at += len;
     *     }
     *   }
     * }
     * assert(0 <= at && at <= n);
     */
    return at;
  }
};