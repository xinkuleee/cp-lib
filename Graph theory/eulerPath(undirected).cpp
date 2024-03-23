vector<PII> g[maxn];
int d[maxn], f[maxn], vis[maxn], idx;
vector<int> vec;

void dfs(int x) {
    while (f[x] < (int)g[x].size()) {
        auto [v, id] = g[x][f[x]];
        f[x]++;
        if (vis[id]) continue;
        vis[id] = 1;
        dfs(v);
    }
    vec.pb(x);
}

bool euler() {
    int st = -1, stn = 0;
    rep(i, 1, n) {
        if (d[i] & 1) stn++, st = i;
    }
    if (!(stn == 0 || (stn == 2 && st != -1))) return false;
    if (st == -1)
        rep(i, 1, n) if (d[i]) { st = i; break; }
    dfs(st);
    // vec.pb(st);
    // reverse(all(vec));
    if ((int)vec.size() != m + 1) return false;
    return true;
}
int main() {
    cin >> n >> m;
    rep(i, 1, m) {
        int u, v;
        cin >> u >> v;
        idx++;
        g[u].pb({v, idx});
        g[v].pb({u, idx});
        d[u]++, d[v]++;
    }
    cout << (euler() ? "Yes" : "No") << '\n';
}