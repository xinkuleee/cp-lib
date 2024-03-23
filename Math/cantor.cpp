ll fac[maxn], A[maxn], w[maxn];
void init(int n) {
    fac[0] = 1;
    rep(i, 1, n) fac[i] = fac[i - 1] * i % mod;
}
ll cantor(int w[], int n) {
    ll ans = 1;
    for (int i = 1; i <= n; i++) { // can optimize by BIT
        for (int j = i + 1; j <= n; j++) {
            if (w[i] > w[j]) A[i]++;
        }
    }
    for (int i = 1; i < n; i++) {
        ans += A[i] * fac[n - i];
    }
    return ans;
}

void decanter(ll x, int n) { // x->rank n->length
    x--;
    vector<int> rest(n, 0);
    iota(rest.begin(), rest.end(), 1); // rest->1,2,3,4...
    for (int i = 1; i <= n; i++) {
        A[i] = x / fac[n - i];
        x %= fac[n - i];
    }
    for (int i = 1; i <= n; i++) {
        w[i] = rest[A[i]];
        rest.erase(lower_bound(rest.begin(), rest.end(), w[i]));
    }
}
