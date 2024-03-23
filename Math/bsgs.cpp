struct Hash_table {
    static const int V = 1000003;
    int fst[V], nxt[V];
    int ctm, ptm[V], T;
    int val[V];
    ll key[V];
    void init() {T = 0, ctm++;}
    void insert(ll k, int v) {
        int s = k % V;
        if (ptm[s] != ctm) ptm[s] = ctm, fst[s] = -1;
        for (int i = fst[s]; i != -1; i = nxt[i]) if (key[i] == k) {
                return;
            }
        nxt[T] = fst[s], fst[s] = T, key[T] = k, val[T] = v;
        T++;
    }
    int query(ll k) {
        int s = k % V;
        if (ptm[s] != ctm) return -1;
        for (int i = fst[s]; i != -1; i = nxt[i]) {
            if (key[i] == k) return val[i];
        }
        return -1;
    }
} hs;

int bsgs(int a, int b, int m) { // a^x=b(mod m)
    int res = m + 1;
    int t = sqrt(m) + 2;
    ll d = powmod(a, t, m);
    ll cnt = 1;
    //map<int,int> p;
    hs.init();
    for (int i = 1; i <= t; i++) {
        cnt = cnt * d % m;
        //if (!p.count(cnt)) p[cnt] = i;
        if (hs.query(cnt) == -1) hs.insert(cnt, i);
    }
    cnt = b;
    for (int i = 1; i <= t; i++) {
        cnt = cnt * a % m;
        //if (p.count(cnt)) res = min(res, p[cnt] * t - i);
        int tmp = hs.query(cnt);
        if (tmp != -1) res = min(res, tmp * t - i);
    }
    if (res >= m) res = -1;
    return res;
}
