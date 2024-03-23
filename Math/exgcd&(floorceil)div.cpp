ll exgcd(ll a, ll b, ll &x, ll &y) { // ax+by=(a,b)
    if (b == 0) {
        x = 1, y = 0;
        return a;
    }
    ll d = exgcd(b, a % b, y, x);
    y -= (a / b) * x;
    return d;
}

ll floordiv(ll a, ll b) {
    if (a % b == 0) return a / b;
    else if ((a > 0 && b > 0) || (a < 0 && b < 0)) return a / b;
    else return a / b - 1;
}

ll ceildiv(ll a, ll b) {
    if (a % b == 0) return a / b;
    else if ((a > 0 && b > 0) || (a < 0 && b < 0)) return a / b + 1;
    else return a / b;
}

void solve() {
    ll a, b, d, l1, l2, r1, r2;
    cin >> a >> b >> d >> l1 >> r1 >> l2 >> r2;
    ll x, y;
    ll g = exgcd(a, b, x, y);
    a /= g, b /= g, d /= g;
    x = d % b * x % b;
    y = (d - a * x) / b;
    /*
    l1<=x=x0+b*t<=r1
    l2<=y=y0-a*t<=r2
    */
    ll L = max(ceildiv(l1 - x, b), ceildiv(r2 - y, -a));
    ll R = min(floordiv(r1 - x, b), floordiv(l2 - y, -a));

    if (L > R) cout << 0 << '\n';
    else cout << R - L + 1 << '\n';
}