void solve(int u, int S) {
  int best = -1, cnt = S + 1;
  auto find_best = [&](auto &find_best, int u, int par) -> void {
    sz[u] = 1, sdom[u] = 0;
    for (auto v : e[u]) {
      if (v == par || del[v]) continue;
      find_best(find_best, v, u);
      sz[u] += sz[v];
      sdom[u] = max(sdom[u], sz[v]);
    }
    sdom[u] = max(sdom[u], S - sz[u]);
    if (sdom[u] < cnt) {
      cnt = sdom[u], best = u;
    }
  };
  find_best(find_best, u, 0);
  int id1 = tot++, dep1 = 0;
  int id2, dep2;
  auto dfs = [&](auto &dfs, int u, int par, int dep) -> void {
    dep1 = max(dep1, dep);
    dep2 = max(dep2, dep);
    Q[u].pb({id1, 1, dep});
    Q[u].pb({id2, -1, dep});
    for (auto v : e[u]) {
      if (v == par || del[v]) continue;
      dfs(dfs, v, u, dep + 1);
    }
  };
  Q[best].pb({id1, 1, 0});
  for (auto v : e[best]) {
    if (del[v]) continue;
    id2 = tot++, dep2 = 0;
    dfs(dfs, v, best, 1);
    fenw[id2] = BIT<ll>(dep2 + 1);
  }
  fenw[id1] = BIT<ll>(dep1 + 1);
  del[best] = 1;
  for (auto v : e[best]) {
    if (!del[v]) solve(v, sz[v]);
  }
}