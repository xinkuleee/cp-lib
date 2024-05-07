template <typename T>
struct BIT {
	vector<T> fenw;
	int n;

	BIT(int _n = 0) : n(_n) {
		fenw.assign(n + 1, 0);
	}

	T query(int x) {
		T v{};
		// while (x >= 0) {
		while (x > 0) {
			v += fenw[x];
			x -= (x & -x);
			// x = (x & (x + 1)) - 1;
		}
		return v;
	}

	void modify(int x, T v) {
		if (x <= 0) return;  // 1-base
		// while (x < n) {
		while (x <= n) {
			fenw[x] += v;
			x += (x & -x);
			// x |= (x + 1);
		}
	}

	int kth(T d) {  // 1-base
		int p = 0;
		T sum{};
		for (int i = 20; i >= 0; i--) {
			if (p + bit(i) <= n && sum + fenw[p + bit(i)] <= d) {
				sum += fenw[p + bit(i)];
				p += bit(i);
			}
		}
		return p;
	}
};