namespace kd {
const int K = 2, N = 2.1e5;
template <typename T>
using P = array<T, K>;
template <typename T>
struct node {
  P<T> pt, mx, mn;
  ll val, sum;
  node *l, *r, *p;
  int id;
  node(const P<T> &_pt = P<T>(), ll _val = 0, int _id = 0)
      : pt(_pt), val(_val), sum(_val), id(_id) {
    mx = mn = pt;
    p = l = r = nullptr;
  }
};
node<ll> *ptr[N];
template <typename T>
void pull(node<T> *u) {
  if (not u) return;
  u->sum = u->val;
  rep(i, 0, K - 1) u->mx[i] = u->mn[i] = u->pt[i];
  if (u->l) {
    u->sum += u->l->sum;
    u->l->p = u;
  }
  if (u->r) {
    u->sum += u->r->sum;
    u->r->p = u;
  }
  rep(i, 0, K - 1) {
    if (u->l) {
      u->mx[i] = max(u->mx[i], u->l->mx[i]);
      u->mn[i] = min(u->mn[i], u->l->mn[i]);
    }
    if (u->r) {
      u->mx[i] = max(u->mx[i], u->r->mx[i]);
      u->mn[i] = min(u->mn[i], u->r->mn[i]);
    }
  }
}

template <typename T>
node<T> *build(vector<node<T>> &a, int l, int r, int d = 0) {
  if (d == K) d = 0;
  if (l >= r) {
    return nullptr;
  } else {
    int md = (l + r) >> 1;
    nth_element(a.begin() + l, a.begin() + md, a.begin() + r,
                [&](node<T> &x, node<T> &y) { return x.pt[d] < y.pt[d]; });
    node<T> *p = new node<T>(a[md]);
    ptr[p->id] = p;
    p->l = build(a, l, md, d + 1);
    p->r = build(a, md + 1, r, d + 1);
    pull(p);
    return p;
  }
}

template <typename T>
node<T> *search(node<T> *u, P<T> p, int d = 0) {
  if (d == K) d = 0;
  if (not u) return nullptr;
  if (u->pt == p) return u;
  if (p[d] < u->pt[d]) {
    return search(u->l, p, d + 1);
  } else if (p[d] > u->pt[d]) {
    return search(u->r, p, d + 1);
  } else {
    auto tmp = search(u->l, p, d + 1);
    if (tmp) return tmp;
    return search(u->r, p, d + 1);
  }
}

template <typename T>
void modify(node<T> *u, ll v) {
  if (not u) return;
  u->val = v;
  for (auto cur = u; cur; cur = cur->p) {
    pull(cur);
  }
}

template <typename T>
bool inside(node<T> *nd, P<T> p, ll c) {
  int cc = 0;
  if (nd->mx[0] * p[0] + nd->mx[1] * p[1] >= c) cc++;
  if (nd->mn[0] * p[0] + nd->mn[1] * p[1] >= c) cc++;
  if (nd->mx[0] * p[0] + nd->mn[1] * p[1] >= c) cc++;
  if (nd->mn[0] * p[0] + nd->mx[1] * p[1] >= c) cc++;
  return cc == 0;
}

template <typename T>
bool outside(node<T> *nd, P<T> p, ll c) {
  int cc = 0;
  if (nd->mx[0] * p[0] + nd->mx[1] * p[1] >= c) cc++;
  if (nd->mn[0] * p[0] + nd->mn[1] * p[1] >= c) cc++;
  if (nd->mx[0] * p[0] + nd->mn[1] * p[1] >= c) cc++;
  if (nd->mn[0] * p[0] + nd->mx[1] * p[1] >= c) cc++;
  return cc == 4;
}

template <typename T>
ll query(node<T> *u, P<T> p, ll c) {
  if (inside(u, p, c)) return u->sum;
  if (outside(u, p, c)) return 0;
  ll s = 0;
  if (u->pt[0] * p[0] + u->pt[1] * p[1] < c) {
    s += u->val;
  }
  if (u->l) s += query(u->l, p, c);
  if (u->r) s += query(u->r, p, c);
  return s;
}

template <typename T>
T eval_min(node<T> *nd,
           P<T> p) {  // 通过估价函数进行启发式搜索，根据当前结果对搜索剪枝
  if (not nd) return numeric_limits<T>::max() / 4;
  ll s = 0;
  rep(i, 0, K - 1) {
    if (p[i] <= nd->mn[i]) s += nd->mn[i] - p[i];
    if (p[i] >= nd->mx[i]) s += p[i] - nd->mx[i];
  }
  return s;
}

template <typename T>
ll mindist(node<T> *u, P<T> p) {
  ll s = numeric_limits<T>::max() / 4;
  if (u->pt != p) {
    s = min(s, abs(u->pt[0] - p[0]) + abs(u->pt[1] - p[1]));
  }
  ll best1 = eval_min(u->l, p), best2 = eval_min(u->r, p);
  if (best1 < best2) {
    if (u->l) s = min(s, mindist(u->l, p));
    if (u->r and best2 < s) s = min(s, mindist(u->r, p));
    return s;
  } else {
    if (u->r) s = min(s, mindist(u->r, p));
    if (u->l and best1 < s) s = min(s, mindist(u->l, p));
    return s;
  }
}

template <typename T>
T eval_max(node<T> *nd,
           P<T> p) {  // 通过估价函数进行启发式搜索，根据当前结果对搜索剪枝
  if (not nd) return 0;
  ll s = 0;
  rep(i, 0, K - 1) s += max(abs(nd->mx[i] - p[i]), abs(nd->mn[i] - p[i]));
  return s;
}

template <typename T>
ll maxdist(node<T> *u, P<T> p) {
  ll s = 0;
  if (u->pt != p) {
    s = max(s, abs(u->pt[0] - p[0]) + abs(u->pt[1] - p[1]));
  }
  ll best1 = eval_max(u->l, p), best2 = eval_max(u->r, p);
  if (best1 > best2) {
    if (u->l) s = max(s, maxdist(u->l, p));
    if (u->r and best2 > s) s = max(s, maxdist(u->r, p));
    return s;
  } else {
    if (u->r) s = max(s, maxdist(u->r, p));
    if (u->l and best1 > s) s = max(s, maxdist(u->l, p));
    return s;
  }
}
}  // namespace kd