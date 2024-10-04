optional<vector<int>> eulerian_path(int n, const vector<PII> &E) {
  vector<int> res;
  if (E.empty()) return res;
  vector<VI> adj(n + 1);
  vector<int> in(n + 1);
  for (int i = 0; i < ssize(E); i++) {
    auto [u, v] = E[i];
    adj[u].push_back(i);
    in[v] += 1;
  }

  int s = -1, hi = 0, lo = 0;
  for (int i = 1; i <= n; i++) {
    if (SZ(adj[i]) == in[i]) continue;
    if (abs(SZ(adj[i]) - in[i]) > 1) return {};
    if (SZ(adj[i]) > in[i]) {
      hi++, s = i;
    } else {
      lo++;
    }
  }
  if (!(hi == 0 && lo == 0) && !(hi == 1 && lo == 1)) {
    return {};
  }
  for (int i = 1; s == -1 && i <= n; i++)
    if (!adj[i].empty()) s = i;

  auto Dfs = [&](auto &Dfs, int u) -> void {
    while (!adj[u].empty()) {
      auto id = adj[u].back();
      adj[u].pop_back();
      int v = E[id].second;
      Dfs(Dfs, v);
      res.push_back(v);
    }
  };
  Dfs(Dfs, s);
  if (SZ(res) != SZ(E)) return {};
  ranges::reverse(res);
  return res;
}
