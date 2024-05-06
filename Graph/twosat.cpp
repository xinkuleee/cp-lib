void dfs(int x) {
    low[x] = dfn[x] = ++idx;
    stk.push(x);
    ins[x] = 1;
    for (auto y : g[x]) {
        if (!dfn[y]) {
            dfs(y);
            low[x] = min(low[x], low[y]);
        } else {
            if (ins[y]) low[x] = min(low[x], dfn[y]);
        }
    }
    if (low[x] >= dfn[x]) {
        ++tot;
        while (true) {
            int cnt = stk.top();
            stk.pop();
            ins[cnt] = 0;
            belong[cnt] = tot;
            if (cnt == x) break;
        }
    }
}
int main() {
    cin >> n >> m;
    char a[10], b[10];
    rep(i, 1, m) {
        cin >> a >> b;
        int t1, t2;
        t1 = a[1] - '0';
        t2 = b[1] - '0';
        int ida0 = (a[0] == 'm' ? t1 : t1 + n);
        int idb0 = (b[0] == 'm' ? t2 : t2 + n);
        int ida1 = (ida0 > n ? ida0 - n : ida0 + n);
        int idb1 = (idb0 > n ? idb0 - n : idb0 + n);
        // a0->b1 b0->a1
        g[idb1].pb(ida0);
        g[ida1].pb(idb0);
    }
    rep(i, 1, 2 * n) if (!dfn[i]) dfs(i);
    bool ok = 1;
    rep(i, 1, n) if (belong[i] == belong[i + n]) {ok = 0; break;}
    cout << (ok ? "GOOD" : "BAD") << '\n';
}