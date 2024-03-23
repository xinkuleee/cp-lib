vector<int> e[maxn], erev[maxn];
vector<int> c, out;
vector<vector<int>> scc;
int vis[maxn];
void dfs(int u) {
    vis[u] = 1;
    for (auto v : e[u]) if (!vis[v]) dfs(v);
    out.pb(u);
}
void dfs_rev(int u) {
    vis[u] = 1;
    for (auto v : erev[u]) if (!vis[v]) dfs_rev(v);
    c.pb(u);
}
void solve() {
    cin >> n >> m;
    rep(i, 1, m) {
        int u, v;
        cin >> u >> v;
        e[u].pb(v);
        erev[v].pb(u);
    }
    rep(i, 1, n) if (!vis[i]) dfs(i);
    fill(vis + 1, vis + n + 1, 0);
    reverse(ALL(out));
    for (auto v : out) if (!vis[v]) {
            c.clear();
            dfs_rev(v);
            scc.pb(c);
        }
}
