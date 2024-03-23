struct BIT {
    ll a[N][N];
    int X, Y;
    BIT(int x, int y): X(x), Y(y) {};
    BIT() {};
    void resize(int x, int y) {
        X = x;
        Y = y;
    }
    ll query(int x, int y) {
        ll res = 0;
        for (int i = x; i > 0; i -= (i & -i))
            for (int j = y; j > 0; j -= (j & -j))
                res += a[i][j];
        return res;
    }
    void modify(int x, int y, ll d) {
        for (int i = x; i <= X; i += (i & -i))
            for (int j = y; j <= Y; j += (j & -j))
                a[i][j] += d;
    }
};
