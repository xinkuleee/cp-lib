ll n, m, k;
vector<int> e[maxn];
int tot, col[maxn];
struct node {
    ll maxv, cnt, l, r;
    node(): maxv(0), l(0), r(0), cnt(0) {}
} seg[maxn * 20];

void upd(int rt) {
    if (seg[seg[rt].l].maxv > seg[seg[rt].r].maxv) {
        seg[rt].maxv = seg[seg[rt].l].maxv;
        seg[rt].cnt = seg[seg[rt].l].cnt;
    } else if (seg[seg[rt].l].maxv < seg[seg[rt].r].maxv) {
        seg[rt].maxv = seg[seg[rt].r].maxv;
        seg[rt].cnt = seg[seg[rt].r].cnt;
    } else {
        seg[rt].maxv = seg[seg[rt].r].maxv;
        seg[rt].cnt = seg[seg[rt].r].cnt + seg[seg[rt].l].cnt;
    }
}

int modify(int rt, int l, int r, int pos) {
    if (rt == 0) rt = ++tot;
    if (l == r) {
        seg[rt].maxv++;
        seg[rt].cnt = pos;
    } else {
        int mid = (l + r) >> 1;
        if (pos <= mid)
            seg[rt].l = modify(seg[rt].l, l, mid, pos);
        else
            seg[rt].r = modify(seg[rt].r, mid + 1, r, pos);
        upd(rt);
    }
    return rt;
}

int merge(int u, int v, int l, int r) {
    if (!u) return v;
    if (!v) return u;
    if (l == r) {
        seg[u].maxv += seg[v].maxv;
        return u;
    } else {
        int mid = (l + r) >> 1;
        seg[u].l = merge(seg[u].l, seg[v].l, l, mid);
        seg[u].r = merge(seg[u].r, seg[v].r, mid + 1, r);
        upd(u);
        return u;
    }
}

ll query(int rt, int l, int r) {
    return seg[rt].cnt;
}

/*void split(int &p, int &q, int s, int t, int l, int r) {
    if (t < l || r < s) return;
    if (!p) return;
    if (l <= s && t <= r) {
        q = p;
        p = 0;
        return;
    }
    if (!q) q = New();
    int m = s + t >> 1;
    if (l <= m) split(ls[p], ls[q], s, m, l, r);
    if (m < r) split(rs[p], rs[q], m + 1, t, l, r);
    push_up(p);
    push_up(q);
}*/

void solve() {
    cin >> n;
    vector<int> rt(n + 1);
    rep(i, 1, n) {
        cin >> col[i];
        rt[i] = modify(0, 1, n, col[i]);
    }
    rep(i, 2, n) {
        int u, v; cin >> u >> v;
        e[u].pb(v), e[v].pb(u);
    }
    vector<ll> ans(n + 1);
    function<void(int, int)> dfs = [&](int u, int f) {
        for (auto v : e[u]) if (v != f) {
                dfs(v, u);
                rt[u] = merge(rt[u], rt[v], 1, n);
            }
        ans[u] = query(rt[u], 1, n);
    };
    dfs(1, 0);
    rep(i, 1, n) cout << ans[i] << " \n"[i == n];
}
