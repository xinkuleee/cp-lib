void solve(int u, int s) {
    int root = -1, cnt = s + 1;
    function<void(int, int)> center = [&](int u, int f) {
        sz[u] = 1, maxs[u] = 0;
        for (auto [v, w] : e[u]) if (v != f && !del[v]) {
                center(v, u);
                sz[u] += sz[v];
                maxs[u] = max(maxs[u], sz[v]);
            }
        maxs[u] = max(maxs[u], s - sz[u]);
        if (maxs[u] < cnt) cnt = maxs[u], root = u;
    };  // using lambda(const auto &self) => faster
    center(u, 0);

    // calc
    vector<pair<int, bool>> d;
    cur[s] = 1;
    function<void(int, int, int)> dfs = [&](int u, int f, int dep) {
        d.pb({dep, cur[s + dep] != 0});
        if (dep == 0 && cur[s] > 1) ans++;
        cur[s + dep]++;
        for (auto [v, w] : e[u]) if (v != f && !del[v]) {
                dfs(v, u, dep + w);
            }
        cur[s + dep]--;
    };  // using lambda(const auto &self) => faster

    for (auto [v, w] : e[root]) if (!del[v]) {
            dfs(v, root, w);
            for (auto [d1, d2] : d) {
                if (d2)
                    ans += c[s - d1][0] + c[s - d1][1];
                else
                    ans += c[s - d1][1];
            }
            for (auto [d1, d2] : d) {
                c[s + d1][d2]++;
            }
            d.clear();
        }
    cur[s]--;
    rep(i, 0, 2 * s) c[i][0] = c[i][1] = 0;

    del[root] = 1;
    for (auto [v, w] : e[root])
        if (!del[v]) solve(v, sz[v]);
}