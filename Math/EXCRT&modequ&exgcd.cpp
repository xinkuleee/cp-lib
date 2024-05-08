ll exgcd(ll a, ll b, ll &x, ll &y) {
    if (b == 0) {
        x = 1, y = 0;
        return a;
    }
    ll d = exgcd(b, a % b, y, x);
    y -= (a / b) * x;
    return d;
}

// 求 a * x = b (mod m) 的解
ll modequ(ll a, ll b, ll m) {
    ll x, y;
    ll d = exgcd(a, m, x, y);
    if (b % d != 0) return -1;
    m /= d; a /= d; b /= d;
    x = x * b % m;
    if (x < 0) x += m;
    return x;
}

void merge(ll &a, ll &b, ll c, ll d) {
    if (a == -1 || b == -1) return;
    ll x, y;
    ll g = exgcd(b, d, x, y);
    if ((c - a) % g != 0) {
        a = -1, b = -1;
        return;
    }
    d /= g;
    ll t = ((c - a) / g) % d * x % d;
    if (t < 0) t += d;
    a = b * t + a;
    b = b * d;
}

