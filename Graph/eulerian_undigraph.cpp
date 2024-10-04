optional<vector<int>> eulerian_path(int n, const vector<PII> &E) {
  vector<int> res;
  if (E.empty()) return res;
  vector<VI> adj(n + 1);
  for (int i = 0; i < ssize(E); i++) {
    auto [u, v] = E[i];
    adj[u].push_back(i);
    adj[v].push_back(i);
  }

  int s = -1, odd = 0;
  for (int i = 1; i <= n; i++) {
    if (ssize(adj[i]) % 2 == 0) continue;
    if (++odd > 2) return {};
    s = i;
  }
  for (int i = 1; s == -1 && i <= n; i++)
    if (!adj[i].empty()) s = i;

  vector<int> vis(ssize(E));
  auto Dfs = [&](auto &Dfs, int u) -> void {
    while (!adj[u].empty()) {
      auto id = adj[u].back();
      adj[u].pop_back();
      if (vis[id]) continue;
      vis[id] = 1;
      int v = u ^ E[id].fi ^ E[id].se;
      Dfs(Dfs, v);
      res.push_back(v);
    }
  };
  Dfs(Dfs, s);
  if (SZ(res) != SZ(E)) return {};
  ranges::reverse(res);
  return res;
}
