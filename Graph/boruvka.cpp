/**
 * while component > 1:
 *     for each component:
 *         find select[i]
 *     for each component:
 *         if select[i] != i:
 *             merge(i, select[i])
 *             component--
 */

ll ans = 0, cnt = n;
while (cnt > 1) {
    fill(select + 1, select + n + 1, -1);
    vector<int> cand;
    for (int i = 1; i <= n; i++) {
        cand.push_back(col[i]);
    }
    ranges::sort(cand);
    cand.erase(unique(all(cand)), cand.end());

    for (auto id : cand) {
        for (auto x : S[id]) remove(x);
        for (auto x : S[id]) {
            auto [opt, w] = get(x);
            if (select[id] == -1 || w < mn[id]) {
                select[id] = opt, mn[id] = w;
            }
        }
        for (auto x : S[id]) insert(x);
    }

    for (int i = 1; i <= n; i++) if (col[i] == i) {
        int j = col[select[i]];
        if (i == j) continue;
        ans += mn[i];
        merge(i, j);
        cnt--;
    }
}