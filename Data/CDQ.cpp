
int ans[maxn], lev[maxn];
array<int, 5> v[maxn], tmp[maxn];

struct BIT {
    ll a[maxn];
    int sz;
    BIT(int x): sz(x) {};
    BIT() {};
    void resize(int x) {
        sz = x;
    }
    ll query(int pos) {
        ll res = 0;
        for (int i = pos; i > 0; i -= (i & -i)) {
            res += a[i];
        }
        return res;
    }
    void modify(int pos, ll d) {
        for (int i = pos; i <= sz; i += (i & -i)) {
            a[i] += d;
        }
    }
} c;

void solve(int l, int r) {
    if (l >= r) return;
    int mid = (l + r) / 2;
    solve(l, mid), solve(mid + 1, r);
    int i = l, j = mid + 1;
    int piv = l;
    while (i <= mid || j <= r) {
        if (i <= mid && (j > r || mp(v[i][1], v[i][2]) <= mp(v[j][1], v[j][2]))) {
            c.modify(v[i][2], v[i][3]);
            tmp[piv++] = v[i++];
        } else {
            v[j][4] += c.query(v[j][2]);
            tmp[piv++] = v[j++];
        }
    }
    rep(i, l, mid) c.modify(v[i][2], -v[i][3]);
    rep(i, l, r) v[i] = tmp[i];
}

void solve() {
    cin >> n >> k;
    c.resize(k);
    rep(i, 1, n) {
        int s, c, m;
        cin >> s >> c >> m;
        v[i] = {s, c, m, 1, 0};
    }
    v[0][0] = -1;
    sort(v + 1, v + n + 1);
    int cnt = 0;
    rep(i, 1, n) {
        if (v[i][0] == v[cnt][0] && v[i][1] == v[cnt][1] && v[i][2] == v[cnt][2]) v[cnt][3]++;
        else v[++cnt] = v[i];
    }
    solve(1, cnt);
    rep(i, 1, cnt) {
        ans[v[i][4] + v[i][3] - 1] += v[i][3];
    }
    rep(i, 0, n - 1) cout << ans[i] << '\n';
}
