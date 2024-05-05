/**
 *    author:  tourist
 *    created: 07.10.2022 20:32:03
**/
#include <bits/stdc++.h>

using namespace std;

#define bit(x) (1ll<<(x))

#ifdef LOCAL
#include "algo/debug.h"
#else
#define debug(...) 42
#endif

mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());

class node {
public:
  int id;
  node* l;
  node* r;
  node* p;
  bool rev;
  int sz;
  // declare extra variables:
  long long P;
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
  pair<node*, int> p = find(v, [&](node * u) {
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

int get_pos(node* v) { // 0-indexed
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

node* get_root(node* v) {
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

pair<node*, node*> split_cnt(node* v, int k) {
  if (v == nullptr) {
    return {nullptr, nullptr};
  }
  v->push();
  int left_and_me = (v->l != nullptr ? v->l->sz : 0) + 1;
  if (k < left_and_me) {
    pair<node*, node*> p = split_cnt(v->l, k);
    if (p.first != nullptr) {
      p.first->p = nullptr;
    }
    v->l = p.second;
    v->pull();
    return {p.first, v};
  } else {
    pair<node*, node*> p = split_cnt(v->r, k - left_and_me);
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

// extra of mine
long long lower(node* u, long long x) {
  if (u == nullptr)
    return numeric_limits<long long>::min();
  else if (x <= u->x)
    return lower(u->l, x);
  else
    return max(u->x, lower(u->r, x));
}

long long upper(node* u, long long x) {
  if (u == nullptr)
    return numeric_limits<long long>::max();
  else if (u->x <= x)
    return upper(u->r, x);
  else
    return min(u->x, upper(u->l, x));
}

}  // namespace treap

using namespace treap;

int n;

int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);
  node* root = nullptr;
  cin >> n;
  for (int i = 1; i <= n; i++) {
    int op;
    long long x;
    cin >> op >> x;
    switch (op) {
      case 1: {
        root = add(root, new node(x, x), [&](node * u) {
          return x < u->x;
        });
        break;
      }
      case 2: {
        auto [pt, w] = find(root, [&](node * u) {
          if (x < u->x) return -1;
          else if (x == u->x) return 0;
          else return 1;
        });
        assert(w == 0);
        root = remove(pt);
        break;
      }
      case 3: {
        cout << count_less(root, x) + 1 << '\n';
        break;
      }
      case 4: {
        cout << get_kth(root, x - 1)->x << '\n';
        break;
      }
      case 5: {
        cout << lower(root, x) << '\n';
        break;
      }
      case 6: {
        cout << upper(root, x) << '\n';
        break;
      }
    }
  }
}