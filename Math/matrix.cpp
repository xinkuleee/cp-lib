struct matrix {
    int r, c;
    vector<vector<ll>> a;
    matrix(int x, int y): r(x), c(y) {
        a = vector<vector<ll>>(r + 1, vector<ll>(c + 1));
    }
    matrix friend operator *(const matrix &x, const matrix &y) {
        matrix res(x.r, y.c);
        assert(x.c == y.r);
        for (int i = 1; i <= res.r; i++)
            for (int j = 1; j <= res.c; j++)
                for (int k = 1; k <= x.c; k++) {
                    res.a[i][j] += x.a[i][k] * y.a[k][j] % mod;
                    if (res.a[i][j] >= mod)
                        res.a[i][j] -= mod;
                }
        return res;
    }
    matrix friend matrixpow(matrix x, ll b) {
        matrix res(x.r, x.c);
        assert(x.r == x.c);
        rep(i, 1, x.r) res.a[i][i] = 1;
        while (b) {
            if (b & 1) res = res * x;
            b >>= 1;
            x = x * x;
        }
        return res;
    }
};
