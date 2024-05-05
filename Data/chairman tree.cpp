
struct node {
    node *l, *r;
    ull val;
};
 
node* build(int l, int r) {
    node* p = new node();
    if (l == r) {
        p->l = p->r = nullptr;
        p->val = 0;
    } else {
        int mid = (l + r) >> 1;
        p->l = build(l, mid);
        p->r = build(mid + 1, r);
        p->val = 0;
    }
    return p;
}
 
ull query(node *v, int l, int r, int ql, int qr) {
    if (ql == l && qr == r) {
        return v->val;
    } else {
        int mid = (l + r) >> 1;
        if (qr <= mid)
            return query(v->l, l, mid, ql, qr);
        else if (ql > mid)
            return query(v->r, mid + 1, r, ql, qr);
        else
            return query(v->l, l, mid, ql, mid) ^ query(v->r, mid + 1, r, mid + 1, qr);
    }
}
 
node* update(node* v, int l, int r, int pos, ull x) {
    if (l == r) {
        node *p = new node();
        p->l = p->r = nullptr;
        p->val = v->val ^ x;
        return p;
    } else {
        int mid = (l + r) >> 1;
        node* p = new node();
        *p = *v;
        if (pos <= mid) p->l = update(v->l, l, mid, pos, x);
        else p->r = update(v->r, mid + 1, r, pos, x);
        p->val = p->l->val ^ p->r->val;
        return p;
    }
}
 

