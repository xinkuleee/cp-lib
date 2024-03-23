vector<int> g[maxn];
int idx;
int a[N][N], use[N][N], p[maxn], vis[maxn];

bool find(int x) {
    vis[x] = 1;
    for (auto y : g[x]) {
        if (!p[y] || (!vis[p[y]] && find(p[y]))) {
            p[y] = x;
            return true;
        }
    }
    return false;
}

int match() {
    int res = 0;
    fill(p + 1, p + idx + 1, 0);
    for (int i = 1; i <= idx; i++) {
        fill(vis + 1, vis + idx + 1, 0);
        if (find(i)) res++;
    }
    return res;
}
