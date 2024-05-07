vector<PII> g[maxn];
stack<int> stk;
int dfn[maxn], ins[maxn], low[maxn];
int idx, tot;
VI ans;
void dfs(int x, int f) {
    low[x] = dfn[x] = ++idx;
    stk.push(x);
    ins[x] = 1;
    for (auto [y, id] : g[x]) {
        if (!dfn[y]) {
            dfs(y, id);
            low[x] = min(low[x], low[y]);
        } else {
            if (ins[y] && id != f) low[x] = min(low[x], dfn[y]);
        }
    }
    if (low[x] >= dfn[x]) {
        ++tot;
        while (true) {
            int cur = stk.top();
            stk.pop();
            ins[cur] = 0;
            if (cur == x) break;
        }
        if (f != 0) ans.pb(f);
    }
}
