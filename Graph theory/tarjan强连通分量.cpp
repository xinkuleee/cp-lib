vector<int> g[maxn];
stack<int> stk;
int dfn[maxn], ins[maxn], low[maxn], belong[maxn];
int idx, tot;

void dfs(int x) {
    low[x] = dfn[x] = ++idx;
    ins[x] = 1;
    stk.push(x);
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
            int cnt = stk.top(); stk.pop();
            ins[cnt] = 0;
            belong[cnt] = tot;
            if (cnt == x) break;
        }
    }
}

