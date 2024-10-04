vector<PII> e[N];

template <typename T>
void add(int u, int v, T w) {
    e[u].eb(v, w);
}

template <typename T>
vector<T> dijkstra(vector<pair<int, T>> *g, int start) {
    // assert(0 <= start && start < g.n);
    // maybe use inf = numeric_limits<T>::max() / 4
    vector<T> dist(n, numeric_limits<T>::max());
    priority_queue<pair<T, int>, vector<pair<T, int>>, greater<pair<T, int>>> s;
    dist[start] = 0;
    s.emplace(dist[start], start);
    while (!s.empty()) {
        T expected = s.top().first;
        int i = s.top().second;
        s.pop();
        if (dist[i] != expected) {
            continue;
        }
        for (auto [to, cost] : g[i]) {
            if (dist[i] + cost < dist[to]) {
                dist[to] = dist[i] + cost;
                s.emplace(dist[to], to);
            }
        }
    }
    return dist;
    // returns numeric_limits<T>::max() if there's no path
}