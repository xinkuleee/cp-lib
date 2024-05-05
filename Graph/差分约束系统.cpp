vector<PII> g[maxn];
int dist[maxn];

int main() {
    cin >> n >> m;
    rep(i, 1, m) {
        int u, v, w;
        cin >> u >> v >> w;
        // xu-xv<=w -> xu<=xv+w
        // xv-xu<=w -> xv<=xu+w
        g[u].pb({v, w});
    }
    fill(dist + 1, dist + 1 + n, 0);
    while (true) {
        bool ok = 0;
        for (int i = 1; i <= n; i++)
            for (auto [v, w] : g[i]) {
                if (dist[v] > dist[i] + w) ok = 1;
                dist[v] = min(dist[v], dist[i] + w);
            }
        if (!ok) break;
    }
    rep(i, 1, n) cout << -dist[i] << '\n';
}