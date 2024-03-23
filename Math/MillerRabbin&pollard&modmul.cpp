/*ModMulLL.h
Description: Calculate a·b mod c (or a
b mod c) for 0 ≤ a, b ≤ c ≤ 7.2·10^18
Time: O (1) for modmul, O (log b) for modpow*/
/*ull modmul(ull a, ull b, ull M) {
    ll ret = a * b - M * ull(1.L / M * a * b);
    return ret + M * (ret < 0) - M * (ret >= (ll)M);
}
ull modpow(ull b, ull e, ull mod) {
    ull ans = 1;
    for (; e; b = modmul(b, b, mod), e /= 2)
        if (e & 1) ans = modmul(ans, b, mod);
    return ans;
}*/
ll modmul(ll a, ll b, ll m) {
    a %= m, b %= m;
    ll d = ((ldb)a * b / m);
    d = a * b - d * m;
    if (d >= m) d -= m;
    if (d < 0) d += m;
    return d;
}
ll modpow(ll a, ll b, ll p) {
    ll ans = 1;
    while (b) {
        if (b & 1) ans = modmul(ans, a, p);
        a = modmul(a, a, p); b >>= 1;
    } return ans;
}
/*MillerRabin.h
Description: Deterministic Miller-Rabin primality test. Guaranteed to
work for numbers up to 7 · 1018; for larger numbers, use Python and extend A randomly.
Time: 7 times the complexity of a^b mod c.*/
bool isPrime(ll n) {
    if (n < 2 || n % 6 % 4 != 1) return (n | 1) == 3;
    ll A[] = {2, 325, 9375, 28178, 450775, 9780504, 1795265022},
              s = __builtin_ctzll(n - 1), d = n >> s;
    for (ll a : A) { // ^ count trailing zeroes
        ll p = modpow(a % n, d, n), i = s;
        while (p != 1 && p != n - 1 && a % n && i--)
            p = modmul(p, p, n);
        if (p != n - 1 && i != s) return 0;
    }
    return 1;
}
/*Factor.h
Description: Pollard-rho randomized factorization algorithm. Returns
prime factors of a number, in arbitrary order (e.g. 2299 -> {11, 19, 11}).
Time: O(n^1/4), less for numbers with small factors.*/
ll pollard(ll n) {
    auto f = [n](ll x) { return modmul(x, x, n) + 1; };
    ll x = 0, y = 0, t = 30, prd = 2, i = 1, q;
    while (t++ % 40 || __gcd(prd, n) == 1) {
        if (x == y) x = ++i, y = f(x);
        if ((q = modmul(prd, max(x, y) - min(x, y), n))) prd = q;
        x = f(x), y = f(f(y));
    }
    return __gcd(prd, n);
}
vector<ll> factor(ll n) {
    if (n == 1) return {};
    if (isPrime(n)) return {n};
    ll x = pollard(n);
    auto l = factor(x), r = factor(n / x);
    l.insert(l.end(), all(r));
    return l;
}
