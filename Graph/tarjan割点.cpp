vector<int> g[maxn], ans;
stack<int> stk;
int dfn[maxn], cut[maxn], low[maxn], idx;

void dfs(int x, int f) {
    low[x] = dfn[x] = ++idx;
    stk.push(x);
    int ch = 0;
    for (auto y : g[x]) {
        if (!dfn[y]) {
            ch++;
            dfs(y, x);
            low[x] = min(low[x], low[y]);
            if (low[y] >= dfn[x]) cut[x] = 1;
        } else {
            if (y != f) low[x] = min(low[x], dfn[y]);
        }
    }
    if (x == 1 && ch <= 1) cut[x] = 0;
    if (cut[x]) ans.pb(x);
}
