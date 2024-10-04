// http://oj.daimayuan.top/course/14/problem/763 欧拉路判断
vector<PII> g[N];
int d[N], f[N], vis[N], edge_idx;
vector<int> path;

void dfs(int x) {
    while (f[x] < SZ(g[x])) {
        auto [v, id] = g[x][f[x]];
        f[x]++;
        if (vis[id]) continue;
        vis[id] = 1;
        dfs(v);
        path.pb(x);
    }
}

bool euler() {
    int start = -1, num = 0;
    rep(i, 1, n) {
        if (d[i] & 1) num++, start = i;
    }
    if (!(num == 0 || (num == 2 && start != -1))) return false;
    if (start == -1) {
        rep(i, 1, n) {
            if (d[i]) {
                start = i;
                break;
            }
        }
    }
    dfs(start);
    path.pb(start);
    reverse(all(path));
    if (SZ(path) != m + 1) return false;
    return true;
}

void solve() {
    cin >> n >> m;
    rep(i, 1, m) {
        int u, v;
        cin >> u >> v;
        edge_idx++;
        g[u].pb({v, edge_idx});
        g[v].pb({u, edge_idx});
        d[u]++, d[v]++;
    }
    cout << (euler() ? "Yes" : "No") << '\n';
}