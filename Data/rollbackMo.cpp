VI rollbackMo(const vector<array<int, 3>> &Q) {
    const int blk = 350;
    vector<VI> s(SZ(Q));
    vector<int> BF, res(SZ(Q));
    for (int i = 0; i < SZ(Q); i++) {
        int u = Q[i][0] / blk, v = Q[i][1] / blk;
        if (u == v) BF.push_back(i);
        else s[u].push_back(i);
    }
    for (int i = 0; i < SZ(Q); i++)
        sort(all(s[i]), [&](int i, int j) { return Q[i][1] < Q[j][1]; });
    for (int qi : BF) {
        for (int i = Q[qi][0]; i <= Q[qi][1]; i++)
            add(i);
        res[qi] = calc(Q[qi][2]);
        for (int i = Q[qi][0]; i <= Q[qi][1]; i++)
            del(i);
    }
    for (const auto &v : s) {
        if (v.empty()) continue;
        int next_blk = (Q[v.back()][0] / blk + 1) * blk;
        int L = next_blk, R = next_blk - 1;
        for (int qi : v) {
            while (R < Q[qi][1]) R++, add(R);
            while (L > Q[qi][0]) L--, add(L);
            res[qi] = calc(Q[qi][2]);
            while (L < next_blk) del(L), L++;
        }
        for (int i = next_blk; i <= R; i++)
            del(i);
    }
    return res;
}
