ll fa[maxn], d[maxn];
void init() {
    rep(i, 1, n) fa[i] = i, d[i] = 0;
}
int find(int x) {
    if (fa[x] == x) return fa[x];
    int p = fa[x];
    fa[x] = find(fa[x]);
    d[x] = d[x] + d[p];
    return fa[x];
}
void unite(int l, int r, ll x) {
    int fl = find(l);
    int fr = find(r);
    fa[fr] = fl;
    d[fr] = d[l] - d[r] + x;
}
