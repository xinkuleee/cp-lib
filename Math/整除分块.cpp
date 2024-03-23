void solve() {
    u64 ans = 0;
    cin >> n;
    for (ll l = 1; l <= n; l++) {
        ll d = n / l, r = n / d;
        ans += (l + r) * (r - l + 1) / 2 * d;
        l = r;
    }
}
