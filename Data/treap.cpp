struct node {
    int l, r;
    int key, val;
    int cnt, size;
} tr[maxn];

int root, idx;

void pushup(int p) {
    tr[p].size = tr[tr[p].l].size + tr[tr[p].r].size + tr[p].cnt;
}

int get_node(int key) {

    tr[ ++ idx].key = key;
    tr[idx].val = mrand();
    tr[idx].cnt = tr[idx].size = 1;
    return idx;
}

void zig(int &p) {   // 右旋

    int q = tr[p].l;
    tr[p].l = tr[q].r, tr[q].r = p, p = q;
    pushup(tr[p].r), pushup(p);
}

void zag(int &p) {   // 左旋

    int q = tr[p].r;
    tr[p].r = tr[q].l, tr[q].l = p, p = q;
    pushup(tr[p].l), pushup(p);
}

void build() {
    get_node(-inf), get_node(inf);
    root = 1, tr[1].r = 2;
    pushup(root);

    if (tr[1].val < tr[2].val) zag(root);
}


void insert(int &p, int key) {
    if (!p) p = get_node(key);
    else if (tr[p].key == key) tr[p].cnt ++ ;
    else if (tr[p].key > key) {
        insert(tr[p].l, key);
        if (tr[tr[p].l].val > tr[p].val) zig(p);
    }
    else {
        insert(tr[p].r, key);
        if (tr[tr[p].r].val > tr[p].val) zag(p);
    }
    pushup(p);
}

void del(int &p, int key) {
    if (!p) return;
    if (tr[p].key == key) {
        if (tr[p].cnt > 1) tr[p].cnt -- ;
        else if (tr[p].l || tr[p].r) {
            if (!tr[p].r || (tr[p].l && (tr[tr[p].l].val > tr[tr[p].r].val))) {
                zig(p);
                del(tr[p].r, key);
            } else {
                zag(p);
                del(tr[p].l, key);
            }
        }
        else p = 0;
    }
    else if (tr[p].key > key) del(tr[p].l, key);
    else del(tr[p].r, key);

    pushup(p);
}

int get_rank(int p, int key) {   // 通过数值找排名

    if (!p) return 0;   // 一般中不会发生此情况
    if (tr[p].key == key) return tr[tr[p].l].size + 1;
    if (tr[p].key > key) return get_rank(tr[p].l, key);
    return tr[tr[p].l].size + tr[p].cnt + get_rank(tr[p].r, key);
}

int get_key(int p, int rank) {  // 通过排名找数值
    if (!p) return inf;     // 一般不会发生此情况
    if (tr[tr[p].l].size >= rank) return get_key(tr[p].l, rank);
    if (tr[tr[p].l].size + tr[p].cnt >= rank) return tr[p].key;
    return get_key(tr[p].r, rank - tr[tr[p].l].size - tr[p].cnt);
}

int get_prev(int p, int key) {  // 找到严格小于key的最大数
    if (!p) return -inf;
    if (tr[p].key >= key) return get_prev(tr[p].l, key);
    return max(tr[p].key, get_prev(tr[p].r, key));
}

int get_next(int p, int key) {   // 找到严格大于key的最小数
    if (!p) return inf;
    if (tr[p].key <= key) return get_next(tr[p].r, key);
    return min(tr[p].key, get_next(tr[p].l, key));
}

