// u * 1 = e, phi * 1 = id, phi = id * u
int n = 1e7 + 15, m1, m2;
int pr[maxn], p[maxn], pe[maxn], u[maxn], tot;
int su[maxn];
int main() {
    p[1] = 1;
    for (int i = 2; i <= n; i++) {
        if (!p[i]) pe[i] = i, p[i] = i, pr[++tot] = i;
        for (int j = 1; j <= tot && pr[j]*i <= n; j++) {
            p[pr[j]*i] = pr[j];
            if (pr[j] == p[i]) {
                pe[pr[j]*i] = pe[i] * p[i];
                break;
            } else {
                pe[pr[j]*i] = pr[j];
            }
        }
    }
    u[1] = 1;
    for (int i = 2; i <= n; i++) {
        if (i == pe[i]) {
            if (i == p[i]) u[i] = -1;
            else u[i] = 0;
        } else {
            u[i] = u[pe[i]] * u[i / pe[i]];
        }
    }
    rep(i, 1, n) su[i] = su[i - 1] + u[i];

    cin >> m1 >> m2;
    ll ans = 0;
    for (int l = 1; l <= m1 && l <= m2; l++) {
        int d1 = m1 / l, d2 = m2 / l;
        int r = min(m1 / d1, m2 / d2);
        ans += 1ll * (m1 / l) * (m2 / l) * (su[r] - su[l - 1]);
        l = r;
    }
    cout << ans << '\n';
}