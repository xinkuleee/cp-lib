/**
 *    author:  tourist
 *    created: 07.10.2022 20:32:03       
**/
#include <bits/stdc++.h>
 
using namespace std;
 
#ifdef LOCAL
#include "algo/debug.h"
#else
#define debug(...) 42
#endif
 
mt19937 rng((unsigned int) chrono::steady_clock::now().time_since_epoch().count());
 
class node {
 public:
  int id;
  node* l;
  node* r;
  node* p;
  bool rev;
  int sz;
  // declare extra variables:
  int P;
  long long add;
  long long x;
 
  node(int _id, long long _x) {
    id = _id;
    l = r = p = nullptr;
    rev = false;
    sz = 1;
    // init extra variables:
    P = rng();
    add = 0;
    x = _x;
  }
 
  // push everything else:
  void push_stuff() {
    if (add != 0) {
      if (l != nullptr) {
        l->unsafe_apply(add);
      }
      if (r != nullptr) {
        r->unsafe_apply(add);
      }
      add = 0;
    }
  }
 
  void unsafe_reverse() {
    push_stuff();
    rev ^= 1;
    swap(l, r);
    pull();
  }
 
  // apply changes:
  void unsafe_apply(long long delta) {
    add += delta;
    x += delta;
  }
 
  void push() {
    if (rev) {
      if (l != nullptr) {
        l->unsafe_reverse();
      }
      if (r != nullptr) {
        r->unsafe_reverse();
      }
      rev = 0;
    }
    push_stuff();
  }
 
  void pull() {
    sz = 1;
    if (l != nullptr) {
      l->p = this;
      sz += l->sz;
    }
    if (r != nullptr) {
      r->p = this;
      sz += r->sz;
    }
  }
};
 
void debug_node(node* v, string pref = "") {
  #ifdef LOCAL
    if (v != nullptr) {
      debug_node(v->r, pref + " ");
      cerr << pref << "-" << " " << v->id << '\n';
      debug_node(v->l, pref + " ");
    } else {
      cerr << pref << "-" << " " << "nullptr" << '\n';
    }
  #endif
}
 
namespace treap {
 
pair<node*, int> find(node* v, const function<int(node*)> &go_to) {
  // go_to returns: 0 -- found; -1 -- go left; 1 -- go right
  // find returns the last vertex on the descent and its go_to
  if (v == nullptr) {
    return {nullptr, 0};
  }
  int dir;
  while (true) {
    v->push();
    dir = go_to(v);
    if (dir == 0) {
      break;
    }
    node* u = (dir == -1 ? v->l : v->r);
    if (u == nullptr) {
      break;
    }
    v = u;
  }
  return {v, dir};
}
 
node* get_leftmost(node* v) {
  return find(v, [&](node*) { return -1; }).first;
}
 
node* get_rightmost(node* v) {
  return find(v, [&](node*) { return 1; }).first;
}
 
node* get_kth(node* v, int k) { // 0-indexed
  pair<node*, int> p = find(v, [&](node* u) {
    if (u->l != nullptr) {
      if (u->l->sz > k) {
        return -1;
      }
      k -= u->l->sz;
    }
    if (k == 0) {
      return 0;
    }
    k--;
    return 1;
  });
  return (p.second == 0 ? p.first : nullptr);
}
 
int get_position(node* v) { // 0-indexed
  int k = (v->l != nullptr ? v->l->sz : 0);
  while (v->p != nullptr) {
    if (v == v->p->r) {
      k++;
      if (v->p->l != nullptr) {
        k += v->p->l->sz;
      }
    }
    v = v->p;
  }
  return k;
}
 
node* get_bst_root(node* v) {
  while (v->p != nullptr) {
    v = v->p;
  }
  return v;
}
 
pair<node*, node*> split(node* v, const function<bool(node*)> &is_right) {
  if (v == nullptr) {
    return {nullptr, nullptr};
  }
  v->push();
  if (is_right(v)) {
    pair<node*, node*> p = split(v->l, is_right);
    if (p.first != nullptr) {
      p.first->p = nullptr;
    }
    v->l = p.second;
    v->pull();
    return {p.first, v};
  } else {
    pair<node*, node*> p = split(v->r, is_right);
    v->r = p.first;
    if (p.second != nullptr) {
      p.second->p = nullptr;
    }
    v->pull();
    return {v, p.second};
  }
}
 
pair<node*, node*> split_leftmost_k(node* v, int k) {
  if (v == nullptr) {
    return {nullptr, nullptr};
  }
  v->push();
  int left_and_me = (v->l != nullptr ? v->l->sz : 0) + 1;
  if (k < left_and_me) {
    pair<node*, node*> p = split_leftmost_k(v->l, k);
    if (p.first != nullptr) {
      p.first->p = nullptr;
    }
    v->l = p.second;
    v->pull();
    return {p.first, v};
  } else {
    pair<node*, node*> p = split_leftmost_k(v->r, k - left_and_me);
    v->r = p.first;
    if (p.second != nullptr) {
      p.second->p = nullptr;
    }
    v->pull();
    return {v, p.second};
  }
}
 
node* merge(node* v, node* u) {
  if (v == nullptr) {
    return u;
  }
  if (u == nullptr) {
    return v;
  }
  if (v->P > u->P) {
//    if (rng() % (v->sz + u->sz) < (unsigned int) v->sz) {
    v->push();
    v->r = merge(v->r, u);
    v->pull();
    return v;
  } else {
    u->push();
    u->l = merge(v, u->l);
    u->pull();
    return u;
  }
}
 
int count_left(node* v, const function<bool(node*)> &is_right) {
  if (v == nullptr) {
    return 0;
  }
  v->push();
  if (is_right(v)) {
    return count_left(v->l, is_right);
  }
  return (v->l != nullptr ? v->l->sz : 0) + 1 + count_left(v->r, is_right);
}
 
int count_less(node* v, long long val) {
  int res = 0;
  while (v != nullptr) {
    v->push();
    if (v->x >= val) {
      v = v->l;
    } else {
      res += (v->l != nullptr ? v->l->sz : 0) + 1;
      v = v->r;
    }
  }
  return res;
}
 
node* add(node* r, node* v, const function<bool(node*)> &go_left) {
  pair<node*, node*> p = split(r, go_left);
  return merge(p.first, merge(v, p.second));
}
 
node* remove(node* v) { // returns the new root
  v->push();
  node* x = v->l;
  node* y = v->r;
  node* p = v->p;
  v->l = v->r = v->p = nullptr;
  v->push();
  v->pull(); // now v might be reusable...
  node* z = merge(x, y);
  if (p == nullptr) {
    if (z != nullptr) {
      z->p = nullptr;
    }
    return z;
  }
  if (p->l == v) {
    p->l = z;
  }
  if (p->r == v) {
    p->r = z;
  }
  while (true) {
    p->push();
    p->pull();
    if (p->p == nullptr) {
      break;
    }
    p = p->p;
  }
  return p;
}
 
node* next(node* v) {
  if (v->r == nullptr) {
    while (v->p != nullptr && v->p->r == v) {
      v = v->p;
    }
    return v->p;
  }
  v->push();
  v = v->r;
  while (v->l != nullptr) {
    v->push();
    v = v->l;
  }
  return v;
}
 
node* prev(node* v) {
  if (v->l == nullptr) {
    while (v->p != nullptr && v->p->l == v) {
      v = v->p;
    }
    return v->p;
  }
  v->push();
  v = v->l;
  while (v->r != nullptr) {
    v->push();
    v = v->r;
  }
  return v;
}
 
int get_size(node* v) {
  return (v != nullptr ? v->sz : 0);
}
 
template<typename... T>
void Apply(node* v, T... args) {
  v->unsafe_apply(args...);
}
 
void reverse(node* v) {
  v->unsafe_reverse();
}
 
}  // namespace treap
 
using namespace treap;
 
int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);
  int n, d, q;
  cin >> n >> d >> q;
  vector<int> a(n);
  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }
  string s;
  cin >> s;
  vector<int> nons;
  for (int i = 0; i < n; i++) {
    if (s[i] == '0') {
      nons.push_back(a[i]);
    }
  }
  int non_ptr = 0;
  vector<int> qk(q), qm(q);
  for (int i = 0; i < q; i++) {
    cin >> qk[i] >> qm[i];
    --qm[i];
  }
  vector<long long> res(q, -1);
  vector<int> order(q);
  iota(order.begin(), order.end(), 0);
  sort(order.begin(), order.end(), [&](int i, int j) {
    return qk[i] < qk[j];
  });
  node* root = nullptr;
  int L = 0;
  int R = -1;
  long long T = 0;
  int q_ptr = 0;
  while (q_ptr < q) {
    if (L > R) {
      R += 1;
      root = merge(root, new node(R, a[R]));
      continue;
    }
    if (qk[order[q_ptr]] == T) {
      int pos = qm[order[q_ptr]];
      res[order[q_ptr]] = (pos < L ? nons[pos] : (pos > R ? a[pos] : get_kth(root, pos - L)->x));
      q_ptr += 1;
      continue;
    }
    long long make = qk[order[q_ptr]] - T;
    long long x = get_leftmost(root)->x;
    if (non_ptr < (int) nons.size() && nons[non_ptr] == x) {
      L += 1;
      root = split_leftmost_k(root, 1).second;
      non_ptr += 1;
      continue;
    }
    long long nxt = (long long) 1e18;
    if (R < n - 1) {
      nxt = a[R + 1];
      if (x + d + (R - L) >= nxt) {
        R += 1;
        root = merge(root, new node(R, a[R]));
        continue;
      }
      if (x + 2 * (d + (R - L)) < nxt) {
        long long full = (nxt - 1 - x) / (d + (R - L)) - 1;
        assert(full > 0);
        make = min(make, full * (R - L + 1));
      } else {
        int cnt = count_less(root, nxt - (d + (R - L)));
        assert(cnt > 0);
        make = min(make, (long long) cnt);
      }
    }
    if (non_ptr < (int) nons.size() && nons[non_ptr] < nxt) {
      int cnt = count_less(root, nons[non_ptr]);
      assert(cnt > 0);
      make = min(make, (long long) cnt);
    }
    assert(make > 0);
    auto rm = make % (R - L + 1);
    auto full = make / (R - L + 1);
    if (rm == 0) {
      Apply(root, full * (d + (R - L)));
    } else {
      auto g3 = split_leftmost_k(root, (int) rm);
      Apply(g3.first, (full + 1) * (d + (R - L)));
      Apply(g3.second, full * (d + (R - L)));
      root = merge(g3.second, g3.first);
    }
    T += make;
  }
  for (int i = 0; i < q; i++) {
    cout << res[i] << '\n';
  }
  return 0;
}