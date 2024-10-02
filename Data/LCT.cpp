namespace linkCutTree {
struct node {
    node *child[2], *parent, *max;
    int id;
    ll sum, val, sz, weight, rev;
    node(ll val, ll weight, int id) : child {nullptr, nullptr}, parent(nullptr), max(this), sum(val), val(val), sz(weight), weight(weight), id(id), rev(false) {}
};

bool isRoot(node *p) { return p->parent == nullptr || (p->parent->child[0] != p && p->parent->child[1] != p); }

int side(node *p) { return p->parent->child[1] == p; }

ll sum(node *p) { return p == nullptr ? 0 : p->sum; }

ll sz(node *p) { return p == nullptr ? 0 : p->sz; }

node *max(node *p) { return p == nullptr ? nullptr : p->max; }

node *max(node *p, node *q) {
    if (p == nullptr)
        return q;
    if (q == nullptr)
        return p;
    return p->weight > q->weight ? p : q;
}

void reverse(node *p) {
    if (p == nullptr)
        return;
    swap(p->child[0], p->child[1]);
    p->rev ^= 1;
}

void push(node *p) {
    if (p->rev == 0)
        return;
    p->rev = 0;
    reverse(p->child[0]);
    reverse(p->child[1]);
}

void pull(node *p) {
    p->sum = sum(p->child[0]) + sum(p->child[1]) + p->val;
    p->max = max(max(max(p->child[0]), max(p->child[1])), p);
    p->sz = p->weight + sz(p->child[0]) + sz(p->child[1]);
}

void connect(node *p, node *q, int side) {
    q->child[side] = p;
    if (p != nullptr)
        p->parent = q;
}

void rotate(node *p) {
    auto q = p->parent;
    int dir = side(p) ^ 1;
    connect(p->child[dir], q, dir ^ 1);
    if (!isRoot(q))
        connect(p, q->parent, side(q));
    else
        p->parent = q->parent;
    connect(q, p, dir);
    pull(q);
}

void splay(node *p) {
    vector<node *> stk;
    for (auto i = p; !isRoot(i); i = i->parent)
        stk.push_back(i->parent);
    while (!stk.empty()) {
        push(stk.back());
        stk.pop_back();
    }
    push(p);
    while (!isRoot(p)) {
        auto q = p->parent;
        if (!isRoot(q))
            rotate(side(p) == side(q) ? q : p);
        rotate(p);
    }
    pull(p);
}

node *access(node *p) {
    node *j = nullptr;
    for (node *i = p; i != nullptr; j = i, i = i->parent) {
        splay(i);
        i->val -= sum(j);
        i->val += sum(i->child[1]);
        i->child[1] = j;
        pull(i);
    }
    splay(p);
    return j;
}

void makeRoot(node *p) {
    access(p);
    reverse(p);
}

void link(node *p, node *q) {
    makeRoot(p);
    access(q);
    p->parent = q;
    q->val += sum(p);
}

void cut(node *p, node *q) {
    makeRoot(p);
    access(q);
    p->parent = q->child[0] = nullptr;
}

node *pathMax(node *p, node *q) {
    makeRoot(p);
    access(q);
    return max(q);
}

ll pathSize(node *p, node *q) {
    makeRoot(p);
    access(q);
    return sz(q);
}

ll rootedSum(node *p) {
    makeRoot(p);
    return sum(p);
}

bool connected(node *p, node *q) {
    access(p);
    access(q);
    return p->parent != nullptr;
}

void fix(node *p, ll v) {
    access(p);
    push(p);
    // modify ...
    p->val += v;
    pull(p);
}

node *lca(node *z,node *x,node *y) {
    makeRoot(z);
    access(x);
    return access(y);
}
}  // namespace linkCutTree
using namespace linkCutTree;
