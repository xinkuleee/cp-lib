ll f[maxn], g[maxn], h[maxn];
int main() {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < bit(n); j++) {
			if ((j & bit(i)) == 0) {
				f[j] += f[j + bit(i)];
				g[j] += g[j + bit(i)];
			}
		}
	}
	for (int i = 0; i < bit(n); i++) {
		f[i] %= mod;
		g[i] %= mod;
		h[i] = f[i] * g[i] % mod;
	}
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < bit(n); j++) {
			if ((j & bit(i)) == 0)
				h[j] -= h[j + bit(i)];
		}
	}
	for (int i = 0; i < bit(n); i++) {
		h[i] %= mod;
		if (h[i] < 0) h[i] += mod;
	}

	ll ans = 0;
	rep(i, 0, bit(n) - 1) ans ^= h[i];
	cout << ans << '\n';
}