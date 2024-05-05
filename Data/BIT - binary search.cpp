struct BIT {
    ll a[maxn];
    int sz;
    BIT(int x): sz(x) {};
    BIT() {};
    void resize(int x) {
        sz = x;
    }
    ll query(ll d) {
        ll res = 0, sum = 0;
        for (int i = 18; i >= 0; i--) {
            if (res + bit(i) <= sz && sum + a[res + bit(i)] <= d) {
                sum += a[res + bit(i)];
                res += bit(i);
            }
        }
        return res;
    }
    void modify(int pos, ll d) {
        for (int i = pos; i <= sz; i += (i & -i)) {
            a[i] += d;
        }
    }
};
      