namespace kd {
const int K = 2, M = 1000005;
const ll inf = 1E16;
extern struct P* null;
struct P {
    ll d[K], l[K], r[K], val;
    ll Max[K], Min[K], sum;
    P *ls, *rs, *fa;
    P* up() {
        rep(i, 0, K - 1) {
            Max[i] = max({d[i], ls->Max[i], rs->Max[i]});
            Min[i] = min({d[i], ls->Min[i], rs->Min[i]});
        }
        sum = val + ls->sum + rs->sum;
        rep(i, 0, K - 1) {
            l[i] = min(d[i], min(ls->l[i], rs->l[i]));
            r[i] = max(d[i], max(ls->r[i], rs->r[i]));
        }
        return ls->fa = rs->fa = this;
    }
} pool[M], *null = new P, *pit = pool;
/*void upd(P* o, int val) {
    o->val = val;
    for (; o != null; o = o->fa)
        o->Max = max(o->Max, val);
}*/
static P *tmp[M], **pt;
void init() {
    null->ls = null->rs = null;
    rep(i, 0, K - 1) null->l[i] = inf, null->r[i] = -inf;
    null->Max[0] = null->Max[1] = -inf;
    null->Min[0] = null->Min[1] = inf;
    null->val = 0;
    null->sum = 0;
}
P* build(P** l, P** r, int d = 0) { // [l, r)
    if (d == K) d = 0;
    if (l >= r) return null;
    P** m = l + (r - l) / 2; assert(l <= m && m < r);
    nth_element(l, m, r, [&](const P * a, const P * b) {
        return a->d[d] < b->d[d];
    });
    P* o = *m;
    o->ls = build(l, m, d + 1); o->rs = build(m + 1, r, d + 1);
    return o->up();
}
P* Build() {
    pt = tmp; for (auto it = pool; it < pit; it++) *pt++ = it;
    P* ret = build(tmp, pt); ret->fa = null;
    return ret;
}
inline bool inside(int p[], int q[], int l[], int r[]) {
    rep(i, 0, K - 1) if (r[i] < q[i] || p[i] < l[i]) return false;
    return true;
}
/*int query(P* o, int l[], int r[]) {
    if (o == null) return 0;
    rep(i, 0, K - 1) if (o->r[i] < l[i] || r[i] < o->l[i]) return 0;
    if (inside(o->l, o->r, l, r)) return o->Max;
    int ret = 0;
    if (o->val > ret && inside(o->d, o->d, l, r)) ret = max(ret, o->val);
    if (o->ls->Max > ret) ret = max(ret, query(o->ls, l, r));
    if (o->rs->Max > ret) ret = max(ret, query(o->rs, l, r));
    return ret;
}
ll eval(P* o, int d[]) { ... }
ll dist(int d1[], int d2[]) { ... }
ll S;
ll query(P* o, int d[]) {
    if (o == null) return 0;
    S = max(S, dist(o->d, d));
    ll mdl = eval(o->ls, d), mdr = eval(o->rs, d);
    if (mdl < mdr) {
        if (S > mdl) S = max(S, query(o->ls, d));
        if (S > mdr) S = max(S, query(o->rs, d));
    } else {
        if (S > mdr) S = max(S, query(o->rs, d));
        if (S > mdl) S = max(S, query(o->ls, d));
    }
    return S;
}*/
bool check(ll x, ll y, ll a, ll b, ll c) { return a * x + b * y < c; }

ll query(P* o, ll a, ll b, ll c) {
    if (o == null) return 0;
    int chk = 0;
    chk += check(o->Min[0], o->Min[1], a, b, c);
    chk += check(o->Max[0], o->Min[1], a, b, c);
    chk += check(o->Min[0], o->Max[1], a, b, c);
    chk += check(o->Max[0], o->Max[1], a, b, c);
    if (chk == 4) return o->sum;
    if (chk == 0) return 0;
    ll ret = 0;
    if (check(o->d[0], o->d[1], a, b, c)) ret += o->val;
    ret += query(o->ls, a, b, c);
    ret += query(o->rs, a, b, c);
    return ret;
}
}  // namespace kd

void solve() {
    cin >> n >> m;
    kd::init();
    rep(i, 1, n) {
        int x, y, w;
        cin >> x >> y >> w;
        kd::pit->d[0] = x, kd::pit->d[1] = y, kd::pit->val = w;
        kd::pit++;
    }
    auto rt = kd::Build();
    rep(i, 1, m) {
        int a, b, c;
        cin >> a >> b >> c;
        cout << kd::query(rt, a, b, c) << '\n';
    }
}