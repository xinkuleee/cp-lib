ll f[N][N];
ll v[N], a[N];
void gauss() {
    for (int i = 1; i <= n; i++) {
        for (int j = i; j <= n; j++) {
            if (f[j][i] > f[i][i]) {
                swap(v[i], v[j]);
                for (int k = 1; k <= n; k++)
                    swap(f[j][k], f[i][k]);
            }
        }
        for (int j = i + 1; j <= n; j++) {
            if (f[j][i]) {
                int delta = f[j][i] * fpow(f[i][i], mod - 2) % mod;
                for (int k = i; k <= n; k++) {
                    f[j][k] -= f[i][k] * delta % mod;
                    if (f[j][k] < 0)
                        f[j][k] += mod;
                }
                v[j] -= v[i] * delta % mod;
                if (v[j] < mod)
                    v[j] += mod;
            }
        }
    }
    for (int j = n; j > 0; j--) {
        for (int k = j + 1; k <= n; k++) {
            v[j] -= f[j][k] * a[k] % mod;
            if (v[j] < 0)
                v[j] += mod;
        }
        a[j] = v[j] * fpow(f[j][j], mod - 2) % mod;
    }
}
