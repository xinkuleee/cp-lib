vector<int> g[maxn];
stack<int> stk;
int dfn[maxn], low[maxn], idx, tot, cut[maxn];
vector<int> bcc[maxn];

void dfs(int x, int f) {
    low[x] = dfn[x] = ++idx;
    stk.push(x);
    int ch = 0;
    for (auto y : g[x]) {
        if (!dfn[y]) {
            ch++;
            dfs(y, x);
            low[x] = min(low[x], low[y]);
            if (low[y] >= dfn[x]) {
                cut[x] = 1;
                ++tot;
                bcc[tot].pb(x);
                while (true) {
                    int cnt = stk.top();
                    stk.pop();
                    bcc[tot].pb(cnt);
                    if (cnt == y) break;
                }
            }
        } else {
            if (y != f) low[x] = min(low[x], dfn[y]);
        }
    }
    if (x == 1 && ch <= 1) cut[x] = 0;
}
