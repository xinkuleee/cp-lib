vector<int> fa(n);
iota(all(fa), 0);
vector<int> sz(n, 1);
vector<pair<int, int>> ops;

auto Get = [&](int i) {
    while (i != fa[i]) {
        i = fa[i];
    }
    return i;
};
auto Unite = [&](int i, int j) {
    i = Get(i), j = Get(j);
    if (i == j) {
        return;
    }
    if (sz[i] > sz[j]) {
        swap(i, j);
    }
    ops.emplace_back(i, fa[i]);
    fa[i] = j;
    ops.emplace_back(~j, sz[j]);
    sz[j] += sz[i];
};
auto RollBack = [&](int T) {
    while (SZ(ops) > T) {
        auto [i, j] = ops.back();
        ops.pop_back();
        if (i >= 0) {
            fa[i] = j;
        } else {
            sz[~i] = j;
        }
    }
};

ll ans = 0;
auto Dfs = [&](auto &&Dfs, int l, int r) -> void {
    if (l == r) {
        for (auto [x, y] : g[l]) {
            x = Get(x);
            y = Get(y);
            ans += 1ll * sz[x] * sz[y];
        }
    } else {
        int mid = midpoint(l, r);
        {
            int save = SZ(ops);
            for (int i = mid + 1; i <= r; i++) {
                for (auto [x, y] : g[i]) {
                    Unite(x, y);
                }
            }
            Dfs(Dfs, l, mid);
            RollBack(save);
        }
        {
            int save = SZ(ops);
            for (int i = l; i <= mid; i++) {
                for (auto [x, y] : g[i]) {
                    Unite(x, y);
                }
            }
            Dfs(Dfs, mid + 1, r);
            RollBack(save);                
        }
    }
};
Dfs(Dfs, 0, n - 1);
