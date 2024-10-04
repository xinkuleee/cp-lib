const int AC_SIGMA = 26, AC_V = 26, AC_N = 810000;
struct AC_automaton {
    struct node {
        node *go[AC_V], *fail, *f;
// declare extra variables:
    } pool[AC_N], *cur, *root, *q[AC_N];
    node* newnode() {
        node *p = cur++;
// init extra variables:
        return p;
    }
// CALL init() and CHECK all const variables:
    void init() { cur = pool; root = newnode(); }
    node* append(node *p, int w) {
        if (!p->go[w]) p->go[w] = newnode(), p->go[w]->f = p;
        return p->go[w];
    }
    void build() {
        int t = 0;
        q[t++] = root;
        root->fail = root;
        rep(i, 0, AC_SIGMA - 1) if (root->go[i]) {
            q[t++] = root->go[i];
            root->go[i]->fail = root;
        } else {
            root->go[i] = root;
        }
        rep(i, 1, t - 1) {
            node *u = q[i];
            rep(j, 0, AC_SIGMA - 1) if (u->go[j]) {
                u->go[j]->fail = u->fail->go[j];
                q[t++] = u->go[j];
            } else {
                u->go[j] = u->fail->go[j];
            }
        }
    }
} ac;
typedef AC_automaton::node ACnode;

const int M = 2, N = 2.1e5;
struct node {
    node *son[M], *go[M], *fail;
    int cnt, vis, ins;
} pool[N], *cur = pool, *q[N], *root;

node *newnode() { return cur++; }
int t, n;

void build() {
    t = 0;
    q[t++] = root;
    for (int i = 0; i < t; i++) {
        node *u = q[i];
        for (int j = 0; j < M; j++) {
            if (u->son[j]) {
                u->go[j] = u->son[j];
                if (u != root)
                    u->go[j]->fail = u->fail->go[j];
                else
                    u->go[j]->fail = root;
                q[t++] = u->son[j];
            } else {
                if (u != root)
                    u->go[j] = u->fail->go[j];
                else
                    u->go[j] = root;
            }
        }
    }
}

void insert(string &s) {
    node *cur = root;
    for (auto c : s) {
        int w = c - '0';
        if (!cur->son[w]) {
            cur->son[w] = newnode();
        }
        cur = cur->son[w];
    }
    cur->cnt = 1;
}