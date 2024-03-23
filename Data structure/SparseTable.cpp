template<class t1>
struct ST {
    int n;
    static const int M = 21;
    t1 p[M][maxn];
    ST() {}
    void build(t1 a[], int sz) {
        n = sz;
        rep(i, 1, n) p[0][i] = a[i];
        rep(i, 1, M - 1)
        rep(j, 1, n) if (j + bit(i) - 1 <= n) {
            p[i][j] = max(p[i - 1][j], p[i - 1][j + bit(i - 1)]);
        }
    }

    t1 query(int l, int r) {
        int len = r - l + 1;
        int k = log2(len);
        return max(p[k][l], p[k][r - bit(k) + 1]);
    }
};
