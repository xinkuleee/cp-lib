struct node {
    ll val;
} seg[maxn << 2];

void update(int id) {
    seg[id].val = max(seg[id * 2].val, seg[id * 2 + 1].val);
}

void build(int l, int r, int id) {
    if (l == r) {
        seg[id].val = a[l];
    } else {
        int mid = (l + r) >> 1;
        build(l, mid, id * 2);
        build(mid + 1, r, id * 2 + 1);
        update(id);
    }
}

void modify(int l, int r, int id, int pos, ll d) {
    if (l == r) {
        seg[id].val = d;
    } else {
        int mid = (l + r) >> 1;
        if (pos <= mid)
            modify(l, mid, id * 2, pos, d);
        else
            modify(mid + 1, r, id * 2 + 1, pos, d);
        update(id);
    }
}


ll search(int l, int r, int id, int ql, int qr, int d) {
    if (ql == l && qr == r) {
        int mid = (l + r) / 2;
        // if (l!=r) pushdown(id); ...
        if (seg[id].val < d) return -1;
        else {
            if (l == r) return l;
            else if (seg[id * 2].val >= d)
                return search(l, mid, id * 2, ql, mid, d);
            else
                return search(mid + 1, r, id * 2 + 1, mid + 1, qr, d);
        }
    } else {
        int mid = (l + r) >> 1;
        // pushdown(id); ...
        if (qr <= mid)
            return search(l, mid, id * 2, ql, qr, d);
        else if (ql > mid)
            return search(mid + 1, r, id * 2 + 1, ql, qr, d);
        else {
            int tmp = search(l, mid, id * 2, ql, mid, d);
            if (tmp != -1)
                return tmp;
            else
                return search(mid + 1, r, id * 2 + 1, mid + 1, qr, d);
        }
    }
}
