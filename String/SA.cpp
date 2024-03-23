struct SA {
    int n, m;
    char s[maxn];
    int sa[maxn], rk[maxn], ht[maxn], x[maxn], y[maxn], c[maxn];
    void init() {
        cin >> (s + 1);
        n = strlen(s + 1), m = 128;
    }
    void get_sa() {
        rep(i, 1, n) c[x[i] = s[i]]++;
        rep(i, 2, m) c[i] += c[i - 1];
        per(i, n, 1) sa[c[x[i]]--] = i;
        // for (int k = 1; k <= n; k <<= 1) {
        for (int k = 1; k < n; k <<= 1) {
            int num = 0;
            rep(i, n - k + 1, n) y[++num] = i;
            rep(i, 1, n) if (sa[i] > k) y[++num] = sa[i] - k;
            rep(i, 1, m) c[i] = 0;
            rep(i, 1, n) c[x[i]]++;
            rep(i, 2, m) c[i] += c[i - 1];
            per(i, n, 1) sa[c[x[y[i]]]--] = y[i], y[i] = 0;
            swap(x, y);
            x[sa[1]] = 1, num = 1;
            rep(i, 2, n)
            x[sa[i]] = (y[sa[i]] == y[sa[i - 1]] && y[sa[i] + k] == y[sa[i - 1] + k]) ? num : ++num;
            if (num == n) break;
            m = num;
        }
    }
    void get_height() {
        rep(i, 1, n) rk[sa[i]] = i;
        for (int i = 1, k = 0; i <= n; i++) {
            if (rk[i] == 1) continue;
            if (k) k--;
            int j = sa[rk[i] - 1];
            while (i + k <= n && j + k <= n && s[i + k] == s[j + k]) k++;
            ht[rk[i]] = k;
        }
    }
};
SA f;
ll fa[maxn], sz[maxn];
vector<array<ll, 3>> seg[maxn];
vector<ll> vec[maxn];
ll len[maxn];

int find(int x) {
    if (fa[x] == x) return fa[x];
    else return fa[x] = find(fa[x]);
}
void init() {
    rep(i, 1, n) fa[i] = i, sz[i] = 1;
}

void answer(int l, int r) {
    rep(i, l, r) cout << f.s[i];
    cout << '\n';
}

void solve() {
    f.init(); f.get_sa(); f.get_height();
    n = f.n;
    init();
    int tp;
    cin >> tp >> k;
    // tp==0 -> 不同位置相同子串算一个
    // tp==1 -> 不同位置相同子串算多个
    rep(i, 1, n) len[i] = n - f.sa[i] + 1;
    if (tp == 0) {
        rep(i, 1, n) {
            if (k > (n - f.sa[i] + 1) - f.ht[i]) k -= (n - f.sa[i] + 1) - f.ht[i];
            else {
                answer(f.sa[i], f.sa[i] + k - 1 + f.ht[i]);
                return;
            }
        } 
        cout << -1 << '\n';
    } else {
        rep(i, 2, n) vec[f.ht[i]].pb(i);
        for (int l = n - 1; l >= 0; l--) {
            for (auto y : vec[l]) {
                int u = find(y - 1), v = find(y);
                if (l < len[u])
                    seg[u].pb({l + 1, len[u], sz[u]});
                if (l < len[v])
                    seg[v].pb({l + 1, len[v], sz[v]});
                fa[v] = u;
                sz[u] += sz[v];
                len[u] = l;
            }
        }
        if (len[1] > 0)
            seg[1].pb({1, len[1], sz[1]});
        rep(i, 1, n) reverse(ALL(seg[i]));
        for (int i = 1; i <= n; i++)
            for (auto [l, r, w] : seg[i]) {
                if (k > (r - l + 1)*w) k -= (r - l + 1) * w;
                else {
                    answer(f.sa[i], f.sa[i] + l - 1 + (k - 1) / w);
                    return;
                }
            }
        cout << -1 << '\n';
    }
}
