vector<PII> g[maxn];
stack<int> stk;
int dfn[maxn], low[maxn], idx, tot, belong[maxn];
vector<int> bcc[maxn];

void dfs(int x, int f) {
    low[x] = dfn[x] = ++idx;
    stk.push(x);
    for (auto [y, id] : g[x]) {
        if (!dfn[y]) {
            dfs(y, id);
            low[x] = min(low[x], low[y]);
        } else {
            if (id != f) low[x] = min(low[x], dfn[y]);
        }
    }
    if (low[x] >= dfn[x]) {
        ++tot;
        while (true) {
            int cur = stk.top();
            stk.pop();
            belong[cur] = tot;
            bcc[tot].pb(cur);
            if (cur == x) break;
        }
    }
}

