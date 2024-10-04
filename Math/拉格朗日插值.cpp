// k阶多项式(需要k+1个点)
// 求在点n上的值
// O(k)
ll lagrange(ll n, int k) {
    vector<ll> x(k + 5), y(k + 5);
    rep(i, 1, k + 1) {
        x[i] = i;
        // y[i]=(y[i-1]+powmod(i,k-1,mod))%mod;
    }
    if (n <= k + 1) return y[n];

    vector<ll> fac(k + 5);
    fac[0] = 1;
    ll coe = 1;
    rep(i, 1, k + 4) fac[i] = fac[i - 1] * i % mod;
    rep(i, 1, k + 1) coe = coe * (n - i + mod) % mod;
    ll ans = 0;
    rep(i, 1, k + 1) {
        ll sgn = (((k + 1 - i) % 2) ? -1 : 1);
        ll f1 = powmod(fac[i - 1] * fac[k + 1 - i] % mod, mod - 2, mod);
        ll f2 = powmod(n - i, mod - 2, mod);
        ans += sgn * coe * f1 % mod * f2 % mod * y[i] % mod;
        ans = (ans + mod) % mod;
    }
    return ans;
}