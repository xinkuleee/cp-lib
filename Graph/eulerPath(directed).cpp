vector<int> g[maxn];
int in[maxn], out[maxn], f[maxn], vis[maxn];
char s[maxn];
vector<int> vec;

void dfs(int x) {
    while (f[x] < (int)g[x].size()) {
        int y = g[x][f[x]];
        f[x]++;
        dfs(y);
    }
    vec.pb(x);
}

bool euler() {
    int st = -1, dif = 0, stn = 0;
    rep(i, 1, n) {
        if (in[i] + 1 == out[i]) stn++, st = i;
        if (in[i] != out[i]) dif++;
    }
    if (!(dif == 0 || (dif == 2 && stn == 1))) return false;
    if (st == -1)
        rep(i, 1, n) if (in[i]) { st = i; break; }
    dfs(st);
    // vec.pb(st);
    // reverse(all(vec));
    if ((int)vec.size() != m + 1) return false;
    return true;
}

int main() {
    cin >> m;
    n = 26;
    rep(i, 1, m) {
        cin >> (s + 1);
        int len = strlen(s + 1);
        int u = s[1] - 'a' + 1, v = s[len] - 'a' + 1;
        g[u].pb(v);
        in[v]++, out[u]++;
    }
    cout << (euler() ? "Yes" : "No") << '\n';
}