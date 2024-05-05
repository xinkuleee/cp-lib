void dfs(int x, int fa) {
    hs[x] = -1, w[x] = 1;
    l[x] = ++tot;
    id[tot] = x;
    for (auto y : g[x]) if (y != fa) {
            dfs(y, x);
            w[x] += w[y];
            if (hs[x] == -1 || w[y] > w[hs[x]])
                hs[x] = y;
        }
    r[x] = tot;
}

void dsu(int x, int fa, int keep) {
    for (auto y : g[x]) {
        if (y != hs[x] && y != fa) {
            dsu(y, x, 0);
        }
    }
    if (hs[x] != -1) dsu(hs[x], x, 1);

    for (auto y : g[x]) {
        if (y != hs[x] && y != fa) {
            for (int i = l[y]; i <= r[y]; i++) {

            }
        }
    }
    // add current node

    ans[x] = cnt;

    if (!keep) {
        // clear
    }
}
