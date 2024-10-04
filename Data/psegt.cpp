struct node {
  node *l, *r;
  ll val, sz, add;
};

void pull(node *u) {
  u->sz = 0, u->val = 0;
  if (u->l) u->sz += u->l->sz, u->val += u->l->val;
  if (u->r) u->sz += u->r->sz, u->val += u->r->val;
}

void push(node *u) {
  if (u->add) {
    if (u->l) {
      node *p = new node();
      *p = *u->l;
      u->l = p;
      p->add += u->add;
      p->val += p->sz * u->add;
    }
    if (u->r) {
      node *p = new node();
      *p = *u->r;
      u->r = p;
      p->add += u->add;
      p->val += p->sz * u->add;
    }
    u->add = 0;
  }
}

node *build(int l, int r) {
  node *p = new node();
  p->add = 0;
  if (l == r) {
    p->l = p->r = nullptr;
    p->val = a[l];
    p->sz = 1;
  } else {
    int mid = (l + r) >> 1;
    p->l = build(l, mid);
    p->r = build(mid + 1, r);
    pull(p);
  }
  return p;
}

ll query(node *v, int l, int r, int ql, int qr) {
  if (ql == l && qr == r) {
    return v->val;
  } else {
    push(v);
    int mid = (l + r) >> 1;
    if (qr <= mid)
      return query(v->l, l, mid, ql, qr);
    else if (ql > mid)
      return query(v->r, mid + 1, r, ql, qr);
    else
      return query(v->l, l, mid, ql, mid) +
             query(v->r, mid + 1, r, mid + 1, qr);
  }
}

node *modify(node *v, int l, int r, int ql, int qr, ll x) {
  if (ql == l && qr == r) {
    node *p = new node();
    *p = *v;
    p->add += x;
    p->val += p->sz * x;
    return p;
  } else {
    push(v);
    int mid = (l + r) >> 1;
    node *p = new node();
    *p = *v;
    if (qr <= mid)
      p->l = modify(v->l, l, mid, ql, qr, x);
    else if (ql > mid)
      p->r = modify(v->r, mid + 1, r, ql, qr, x);
    else
      p->l = modify(v->l, l, mid, ql, mid, x),
      p->r = modify(v->r, mid + 1, r, mid + 1, qr, x);
    pull(p);
    return p;
  }
}
