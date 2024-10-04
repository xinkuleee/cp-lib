// http://oj.daimayuan.top/course/14/problem/765 单词接龙
vector<int> g[N];
int in[N], out[N], f[N], vis[N];
string s;
vector<int> path;

void dfs(int x) {
    while (f[x] < SZ(g[x])) {
        int y = g[x][f[x]];
        f[x]++;
        dfs(y);
        path.pb(x);
    }
}

bool euler() {
    int start = -1, diff = 0, num = 0;
    rep(i, 0, n - 1) {
        if (in[i] + 1 == out[i]) num++, start = i;
        if (in[i] != out[i]) diff++;
    }
    // 恰好都balance或者恰好一个in = out + 1，一个in + 1 = out
    if (!(diff == 0 || (diff == 2 && num == 1))) return false;
    if (start == -1) {
        rep(i, 0, n - 1) {
            if (in[i]) {
                start = i;
                break;
            }
        }
    }
    dfs(start);
    path.pb(start);
    reverse(all(path));
    if (SZ(path) != m + 1) return false;
    return true;
}

void solve() {
    cin >> m;
    n = 26;
    rep(i, 1, m) {
        cin >> s;
        int u = s[0] - 'a', v = s[SZ(s) - 1] - 'a';
        g[u].pb(v);
        in[v]++, out[u]++;
    }
    cout << (euler() ? "Yes" : "No") << '\n';
}