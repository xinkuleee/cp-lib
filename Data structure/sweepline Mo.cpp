int main() {
    std::ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);

    for (int i = 1; i <= m; i++) {
        int x, y;
        cin >> x >> y;
        q.pb({x, y, i});
        rej[i] = (y - x + 1LL) * (y - x) / 2LL;
    }
    sort(q.begin(), q.end(), [&](array<int, 3> a, array<int, 3> b)->bool{
        if (getb(a[0]) == getb(b[0]))
            if (getb(a[0]) & 1)
                return a[1] < b[1];
            else
                return a[1] > b[1];
        else return getb(a[0]) < getb(b[0]);
    });

    int L = 1, R = 0;
    for (int i = 0; i < m; i++) {
        while (R < q[i][1]) R++, add(R);
        while (L > q[i][0]) L--, add(L);
        while (L < q[i][0]) del(L), L++;
        while (R > q[i][1]) del(R), R--;
        ans[q[i][2]] = tmp;
    }
}