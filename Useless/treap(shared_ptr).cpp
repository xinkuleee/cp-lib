#include <bits/stdc++.h>
using namespace std;

struct Tree {
    std::shared_ptr<Tree> l;
    std::shared_ptr<Tree> r;
    // Tree *l, *r;
    int v;
    int s;
    Tree(int v = -1) : l(nullptr), r(nullptr), v(v), s(1) {}
    void pull() {
        s = 1;
        if (l != nullptr) {
            s += l->s;
        }
        if (r != nullptr) {
            s += r->s;
        }
    }
};

using pTree = std::shared_ptr<Tree>;
// using pTree = Tree*;

std::pair<pTree, pTree> split(pTree t, int k) {
    if (k == 0) {
        return {nullptr, t};
    }
    if (k == t->s) {
        return {t, nullptr};
    }
    pTree nt = std::make_shared<Tree>();
    // pTree nt = new Tree();
    *nt = *t;
    if (t->l != nullptr && k <= t->l->s) {
        auto [a, b] = split(t->l, k);
        nt->l = b;
        nt->pull();
        return {a, nt};
    } else {
        auto [a, b] = split(t->r, k - 1 - (t->l == nullptr ? 0 : t->l->s));
        nt->r = a;
        nt->pull();
        return {nt, b};
    }
}

std::tuple<pTree, pTree, pTree> split3(pTree t, int l, int r) {
    auto [LM, R] = split(t, r);
    auto [L, M] = split(LM, l);
    return {L, M, R};
}

std::mt19937 rnd(std::chrono::steady_clock::now().time_since_epoch().count());

pTree merge(pTree a, pTree b) {
    if (a == nullptr) {
        return b;
    }
    if (b == nullptr) {
        return a;
    }
    
    pTree t = std::make_shared<Tree>();
    // pTree t = new Tree();

    if (int(rnd() % (a->s + b->s)) < a->s) {
        *t = *a;
        t->r = merge(a->r, b);
    } else {
        *t = *b;
        t->l = merge(a, t->l);
    }
    t->pull();
    
    return t;
}

pTree build(const std::vector<int> &v, int l, int r) {
    if (l == r) {
        return nullptr;
    }
    int m = (l + r) / 2;
    auto t = std::make_shared<Tree>(v[m]);
    // auto t = new Tree(v[m]);
    
    t->l = build(v, l, m);
    t->r = build(v, m + 1, r);
    t->pull();
    return t;
}

void rec(pTree t, std::vector<int> &v, int &cnt) {
    if (t == nullptr) {
        return;
    }
    rec(t->l, v, cnt);
    v[cnt++] = t->v;
    rec(t->r, v, cnt);
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    
    int n;
    std::cin >> n;
    
    std::vector<int> a(n);
    for (int i = 0; i < n; i++) {
        std::cin >> a[i];
    }
    
    std::vector<std::vector<int>> b{a};
    while (!std::is_sorted(b.back().begin(), b.back().end(), std::greater())) {
        auto v = b.back();
        
        std::vector<int> f(n + 1), g(n + 1);
        for (int i = 1, j = 0, k = 0; i <= n; ) {
            if ((i - j) % (i - k) == 0) {
                f[i] = k;
                g[i] = j;
            } else {
                f[i] = f[k] + i - k;
                g[i] = g[k] + i - k;
            }
            if (i == n || v[i] < v[k]) {
                while (j <= k) {
                    j += i - k;
                }
                k = j;
                i = k + 1;
            } else if (v[i] == v[k]) {
                i++;
                k++;
            } else {
                k = j;
                i++;
            }
        }
        
        auto t = build(v, 0, n);
        
        for (int i = n - 1, j = -1; i >= 0; i--) {
            int x = j == i + 1 ? f[i + 1] : g[i + 1];
            j = x;
            auto [l, m, r] = split3(t, x, x + n - i);
            auto [a, b] = split(t, i);
            t = merge(a, m);
        }
        
        int cnt = 0;
        rec(t, v, cnt);
        b.push_back(v);
    }
    
    int q;
    std::cin >> q;
    
    while (q--) {
        int i, j;
        std::cin >> i >> j;
        j--;
        i = std::min(i, int(b.size()) - 1);
        std::cout << b[i][j] << "\n";
    }
    
    return 0;
}
