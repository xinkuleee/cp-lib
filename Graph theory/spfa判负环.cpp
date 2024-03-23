bool spfa() {
    queue<int> q; // dist初值不影响负环的判断，存在负环即会一直更新
    rep(i, 1, n) {
        q.push(i);
        ins[i] = 1;
        num[i] = 0;
    }
    while (q.size()) {
        int p = q.front();
        q.pop();
        ins[p] = 0;
        for (auto [v, w] : g[p]) {
            if (dist[v] > dist[p] + w) {
                dist[v] = dist[p] + w;
                num[v] = num[p] + 1;
                if (num[v] >= n) return true;
                if (!ins[v]) {
                    ins[v] = 1;
                    q.push(v);
                }
            }
        }
    }
    return false;
}
