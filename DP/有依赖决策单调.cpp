pair<int, int> stk[N];
auto calc = [&](int i, int j) { ... } // dp[j] -> dp[i]
int h = 0, t = 0;
stk[t++] = {1, 0}; // {left, opt}

for (int i = 1; i <= n; i++) {
    if (h < t && stk[h].first < i) stk[h].first++;
    if (h + 1 < t && stk[h].first >= stk[h + 1].first) ++h;
    dp[i] = calc(i, stk[h].second);
    while (h < t && calc(stk[t - 1].first, stk[t - 1].second) >= calc(stk[t - 1].first, i))
        --t;
    if (h < t) {
        int l = stk[t - 1].first, r = n + 1;
        while (l + 1 < r) {
            int md = (l + r) >> 1;
            if (calc(md, stk[t - 1].second) < calc(md, i)) l = md; else r = md;
        }
        if (r <= n) stk[t++] = {r, i};
    } else stk[t++] = {i, i};
}