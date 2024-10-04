vector<Mint> fact(1, 1);
vector<Mint> inv_fact(1, 1);

Mint C(int n, int k) {
    if (k < 0 || k > n) {
        return 0;
    }
    while ((int)fact.size() < n + 1) {
        fact.push_back(fact.back() * (int)fact.size());
        inv_fact.push_back(1 / fact.back());
    }
    return fact[n] * inv_fact[k] * inv_fact[n - k];
}

const int mod = 1000000007;
const int T = 1000000;
ll fact[] = {};
ll powmod(ll a, ll b) {
    ll ret = 1;
    for (; b; b >>= 1) {
        if (b & 1) ret = ret * a % mod;
        a = a * a %mod;
    }
    return ret;
}
ll fac(int n) {
    ll v = fact[n / T];
    for (int i = n / T * T + 1; i <= n; i++)
        v = v * i % mod;
    return v;
}
ll binom(int n, int m) {
    if (m < 0 || m > n) return 0;
    return fac(n) * powmod(fac(m) * fac(n - m) % mod, mod - 2) % mod;
}