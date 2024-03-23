ll fac[maxn], fnv[maxn];

ll binom(ll a, ll b) {
    if (b > a || b < 0) return 0;
    return fac[a] * fnv[a - b] % p * fnv[b] % p;
}

ll lucas(ll a, ll b, ll p) {
    ll ans = 1;
    while (a > 0 || b > 0) {
        ans = (ans * binom(a % p, b % p)) % p;
        a /= p, b /= p;
    }
    return ans;
}

int main() {
    cin >> p >> T;
    fac[0] = 1;
    rep(i, 1, p - 1) fac[i] = fac[i - 1] * i % p;
    fnv[p - 1] = powmod(fac[p - 1], p - 2, p);
    per(i, p - 2, 0) fnv[i] = fnv[i + 1] * (i + 1) % p;
    assert(fnv[0] == 1);
}