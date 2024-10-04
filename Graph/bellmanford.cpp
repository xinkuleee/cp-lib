vector<PII> e[N];

template <typename T>
void add(int u, int v, T w) {
    e[u].eb(v, w);
}

template <typename T>
vector<T> bellmanford(vector<pair<int, T>> *g, int start) {
    // assert(0 <= start && start < g.n);
    // maybe use inf = numeric_limits<T>::max() / 4
    const T inf = numeric_limits<T>::max() / 4;
    vector<T> dist(n, inf);
    dist[start] = 0;
    int cnt = 0;
    while (true) {
        bool upd = 0;
        cnt++;
        for (int i = 0; i < n; i++) {
            for (auto [to, cost] : e[i]) {
                if (dist[to] > dist[i] + cost) {
                    upd = 1;
                    dist[to] = dist[i] + cost;
                }
            }
        }
        if (!upd || cnt == n) {
            break;
        }
    }
    return dist;
    // returns inf if there's no path
}
