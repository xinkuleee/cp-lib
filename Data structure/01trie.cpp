struct node {
    int son[2];
    int end;
    int sz;
} seg[maxn << 2];
int root, tot;
int n, m;

void insert(ll x) {
    int cnt = root;
    for (int i = 62; i >= 0; i--) {
        int w = (x >> i) & 1;
        if (seg[cnt].son[w] == 0) seg[cnt].son[w] = ++tot;
        cnt = seg[cnt].son[w];
        seg[cnt].sz++;
    }
    seg[cnt].end++;
}

ll query(ll x, ll k) {
    ll res = 0;
    int cnt = root;
    for (int i = 62; i >= 0; i--) {
        int w = (x >> i) & 1;
        if (seg[seg[cnt].son[w]].sz >= k) cnt = seg[cnt].son[w];
        else {
            k -= seg[seg[cnt].son[w]].sz;
            cnt = seg[cnt].son[abs(w - 1)];
            res += bit(i);
        }
    }
    return res;
}
