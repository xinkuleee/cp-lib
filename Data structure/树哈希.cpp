basic_string<int> e[maxn];
ull hashv[maxn];
ull seed1, seed2, seed3, seed4;

ull f(ull x) { return x * x * x * seed1 + x * seed2; }
ull h(ull x) { return f(x) ^ ((x & seed3) >> 31) ^ ((x & seed4) << 31); }

void dfs1(int u, int fa) {
    hashv[u] = 1;
    for (auto v : e[u]) if (v != fa) {
            dfs1(v, u);
            hashv[u] += h(hashv[v]);
        }
}

void dfs2(int u, int fa, ull fv) {
// for each root
    hashv[u] += fv;
    for (auto v : e[u]) if (v != fa) {
            dfs2(v, u, h(hashv[u] - h(hashv[v])));
        }
}

void solve() {
    seed1 = rng(), seed2 = rng();
    seed3 = rng(), seed4 = rng();
    cin >> n;
    rep(i, 2, n) {
        int u, v;
        cin >> u >> v;
        e[u].pb(v);
        e[v].pb(u);
    }
    dfs1(1, 0);
    sort(hashv + 1, hashv + n + 1);
    n = unique(hashv + 1, hashv + n + 1) - hashv - 1;
    cout << n << '\n';
}