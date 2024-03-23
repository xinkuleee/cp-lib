int n, m, k;
vector<PII> g[maxn];
int dist[maxn];
bool ins[maxn];
void dijkstra(int st, int end) {
    fill(dist + 1, dist + n + 1, bit(29));
    fill(ins + 1, ins + n + 1, 0);
    dist[st] = 0;
    while (true) {
        int cnt = -1;
        bool ok = 0;
        for (int i = 1; i <= n; i++) {
            if (dist[i] < bit(28) && !ins[i])
                if (cnt == -1 || dist[i] < dist[cnt])
                    cnt = i, ok = 1;
        }
        if (!ok) break;
        ins[cnt] = 1;
        for (auto [id, w] : g[cnt]) {
            dist[id] = min(dist[id], dist[cnt] + w);
        }
    }
}

