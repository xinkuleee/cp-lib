vector<PII> e[N];

template <typename T>
void add(int u, int v, T w) {
    e[u].eb(v, w);
}

template <typename T>
vector<T> dijkstra(vector<pair<int, T>> *g, int start) {
    // assert(0 <= start && start < g.n);
    // maybe use inf = numeric_limits<T>::max() / 4
    const T inf = numeric_limits<T>::max();
    vector<T> dist(n, inf);
    vector<int> was(n, 0);
    dist[start] = 0;
    while (true) {
        int cur = -1;
        for (int i = 0; i < n; i++) {
            if (was[i] || dist[i] == inf) continue;
            if (cur == -1 || dist[i] < dist[cur]) {
                cur = i;
            }
        }
        if (cur == -1 || dist[cur] == inf) {
            break;
        }
        was[cur] = 1;
        for (auto [to, cost] : g[cur]) {
            dist[to] = min(dist[to], dist[cur] + cost);
        }
    }
    return dist;
    // returns inf if there's no path
}