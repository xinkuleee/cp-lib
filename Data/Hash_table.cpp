struct Hash_table {
    static const int V = 1000003;
    int fst[V], nxt[V];
    int ctm, ptm[V], T;
    int val[V];
    ll key[V];
    void init() {T = 0, ctm++;}
    void insert(ll k, int v) {
        int s = k % V;
        if (ptm[s] != ctm) ptm[s] = ctm, fst[s] = -1;
        for (int i = fst[s]; i != -1; i = nxt[i]) if (key[i] == k) {
                return;
            }
        nxt[T] = fst[s], fst[s] = T, key[T] = k, val[T] = v;
        T++;
    }
    int query(ll k) {
        int s = k % V;
        if (ptm[s] != ctm) return -1;
        for (int i = fst[s]; i != -1; i = nxt[i]) {
            if (key[i] == k) return val[i];
        }
        return -1;
    }
};