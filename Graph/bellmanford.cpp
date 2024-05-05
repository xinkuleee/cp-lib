vector<PII> g[maxn];
int dist[maxn];
bool ins[maxn];
void bellman_ford(int st) {
    memset(dist, 0x3f, sizeof dist);
    memset(ins, 0, sizeof ins);
    dist[st] = 0;
    int cnt = 0;
    while (true) {
        cnt++;
        bool ok = 0;
        for (int i = 1; i <= n; i++)
            for (auto [v, w] : g[i]) {
                if (dist[v] > dist[i] + w) {
                    dist[v] = dist[i] + w;
                    ok = 1;
                }
            }
        if (!ok) break;
        if (cnt == n) break;
    }
}

