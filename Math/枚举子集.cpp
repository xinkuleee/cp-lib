void solve() {
    f[0] = 1;
    for (int i = 1; i < (1ll << n); i++) {
        int t = i;
        ll res = 0;
        while (true) {
            if (t == 0) break;
            t = (t - 1)&i;
            res = (res + f[t]) % mod;
        }
        f[i] = res * i;
    }
}
