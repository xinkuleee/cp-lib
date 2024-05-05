
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
};
