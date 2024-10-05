/**
 * #define K(x) pii(x.first/blk, x.second ^ -(x.first/blk & 1))
 * 	iota(all(s), 0);
 * 	sort(all(s), [&](int s, int t){ return K(Q[s]) < K(Q[t]); });
 */

VI Mo(const vector<array<int, 3>> &Q) {
    const int blk = 350;
    vector<int> s(SZ(Q)), res = s;
    iota(all(s), 0);
    sort(all(s), [&](int i, int j) {
        int u = Q[i][0] / blk, v = Q[j][0] / blk;
        return u == v ? u % 2 ? Q[i][1] > Q[j][1] : Q[i][1] < Q[j][1] : u < v;
    });
    int L = 1, R = 0;
    for (int qi : s) {
        while (R < Q[qi][1]) R++, add(R);
        while (L > Q[qi][0]) L--, add(L);
        while (R > Q[qi][1]) del(R), R--;
        while (L < Q[qi][0]) del(L), L++;
        res[qi] = calc(Q[qi][2]);
    }
    return res;
}
