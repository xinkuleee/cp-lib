
vector<PII> g[maxn];
int dist[maxn], use[maxn];
int prim(int st) {
    fill(dist + 1, dist + n + 1, bit(29));
    fill(use + 1, use + n + 1, 0);
    dist[st] = 0;
    priority_queue<PII, vector<PII>, greater<PII>> heap;
    heap.push({0, st});
    int res = 0;
    while (heap.size()) {
        auto [w, id] = heap.top();
        heap.pop();
        if (use[id]) continue;
        use[id] = 1;
        res += dist[id];
        for (auto [v, w1] : g[id]) {
            if (dist[v] > w1) {
                dist[v] = w1;
                heap.push({dist[v], v});
            }
        }
    }
    return res;
}

