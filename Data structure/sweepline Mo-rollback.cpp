int n, q, k, block;
int cnt[maxn], ans[maxn], a[maxn], vis[maxn];
vector<array<int, 4>> que;

int getb(int x) {
    return (x - 1) / block + 1;
}

int main() {
    std::ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);

    cin >> n;
    block = sqrt(n);

    rep(i, 1, n) cin >> a[i];
    cin >> q;
    rep(i, 1, q) {
        int l, r;
        cin >> l >> r >> k;
        que.pb({l, r, i, k});
    }
    sort(ALL(que), [&](array<int, 4> a, array<int, 4> b)->bool{
        if (getb(a[0]) != getb(b[0]))
            return getb(a[0]) < getb(b[0]);
        else
            return a[1] < b[1];
    });

    int len = que.size();
    int l, r;
    auto add = [&](int x, int t) {
        cnt[vis[a[x]]]--;
        vis[a[x]]++;
        cnt[vis[a[x]]]++;
    };
    auto del = [&](int x) {
        cnt[vis[a[x]]]--;
        vis[a[x]]--;
        cnt[vis[a[x]]]++;
    };

    for (int x = 0; x < len;) {
        int y = x;
        while (y < len && getb(que[y][0]) == getb(que[x][0])) y++;
        //暴力块内
        while (x < y && que[x][1] <= getb(que[x][0])*block) {
            for (int j = que[x][0]; j <= que[x][1]; j++)
                add(j, que[x][3]);
            ans[que[x][2]] = cnt[que[x][3]];
            for (int j = que[x][0]; j <= que[x][1]; j++)
                del(j);
            x++;
        }
        //块外
        r = getb(que[x][0]) * block;
        while (x < y) {
            l = getb(que[x][0]) * block + 1;
            while (r < que[x][1]) r++, add(r, que[x][3]);
            while (l > que[x][0]) l--, add(l, que[x][3]);
            ans[que[x][2]] = cnt[que[x][3]];
            for (int j = que[x][0]; j <= getb(que[x][0])*block; j++)
                del(j);
            x++;
        }
        for (int j = getb(que[x - 1][0]) * block + 1; j <= que[x - 1][1]; j++)
            del(j);
    }
    rep(i, 1, q) cout << ans[i] << '\n';
}
