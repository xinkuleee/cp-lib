struct linear_base {
    ll w[64];
    ll zero = 0;
    ll tot = -1;
    void clear() {
        rep(i, 0, 63) w[i] = 0;
        zero = 0;
        tot = -1;
    }
    void insert(ll x) {
        for (int i = 62; i >= 0; i--) {
            if (x & bit(i))
                if (!w[i]) {w[i] = x; return;}
                else x ^= w[i];
        }
        zero++;
    }
    void build() {
        rep(i, 0, 63) rep(j, 0, i - 1) {
            if (w[i]&bit(j)) w[i] ^= w[j];
        }
        for (int i = 0; i <= 62; i++) {
            if (w[i] != 0) w[++tot] = w[i];
        }
    }
    ll qmax() {
        ll res = 0;
        for (int i = 62; i >= 0; i--) {
            res = max(res, res ^ w[i]);
        }
        return res;
    }
    bool check(ll x) {
        for (int i = 62; i >= 0; i--) {
            if (x & bit(i))
                if (!w[i]) return false;
                else x ^= w[i];
        }
        return true;
    }
    ll query(ll k) {
        ll res = 0;
        // if (zero) k-=1;
        // if (k >= bit(tot)) return -1;
        for (int i = tot; i >= 0; i--) {
            if (k & bit(i)) {
                res = max(res, res ^ w[i]);
            } else {
                res = min(res, res ^ w[i]);
            }
        }
        return res;
    }
};