int a[maxn], l[maxn], r[maxn], root;
int ans[maxn], tot;

void build() {
    stack<int> stk;
    for (int i = 1; i <= n; i++) {
        int last = 0;
        while (!stk.empty() && a[stk.top()] > a[i]) {
            last = stk.top();
            stk.pop();
        }
        if (stk.empty())
            root = i;
        else
            r[stk.top()] = i;
        l[i] = last;
        stk.push(i);
    }
}

void dfs(int c, int L, int R) {
    ans[c] = ++tot;
    if (l[c]) dfs(l[c], L, c - 1);
    if (r[c]) dfs(r[c], c + 1, R);
}

