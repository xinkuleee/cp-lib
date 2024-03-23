array<ll, 3> a[maxn];
int q[maxn];
ll ans[maxn];

ll X(int p) {
    return 2ll * a[p][0];
}
ll Y(int p) {
    return a[p][0] * a[p][0] + a[p][1];
}
ldb slope(int x, int y) {
    return (ldb)(Y(y) - Y(x)) / (X(y) - X(x));
}
void solve() {
    cin >> n;
    int head = 1, rear = 0;
    rep(i, 1, n) {
        cin >> a[i][0] >> a[i][1];
        a[i][2] = i;
    }
    sort(a + 1, a + n + 1);

    rep(i, 1, n) {
        while (head < rear && slope(q[rear], i) <= slope(q[rear], q[rear - 1])) rear--;
        q[++rear] = i;
    }
    rep(i, 1, n) {
        ll k = -a[i][0];
        while (head < rear && slope(q[head], q[head + 1]) <= k) head++;
        ans[a[i][2]] = (a[i][0] + a[q[head]][0]) * (a[i][0] + a[q[head]][0]) + a[i][1] + a[q[head]][1];
    }
    rep(i, 1, n) cout << ans[i] << '\n';
}

