int n, m, k;
vector<PII> g[maxn];
int dist[maxn];
bool use[maxn];
void dijkstra(int st, int end) {
    fill(dist + 1, dist + n + 1, bit(29));
    fill(use + 1, use + n + 1, 0);
    dist[st] = 0;
    priority_queue<PII, vector<PII>, greater<PII>> heap;
    heap.push({0, st});
    while (heap.size()) {
        auto [w, id] = heap.top();
        heap.pop();
        if (id == end) break;
        if (use[id]) continue;
        use[id] = 1;
        for (auto [v, w1] : g[id]) {
            if (dist[v] > dist[id] + w1) {
                dist[v] = dist[id] + w1;
                heap.push({dist[v], v});
            }
        }
    }
}

