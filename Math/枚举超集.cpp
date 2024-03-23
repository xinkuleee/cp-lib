void solve() {
    for (int i = 1; i < (1ll << n); i++) {
        int t = i;
        while (true) {
            t = (t + 1) | i;
            if (t == bit(n) - 1) break;
        }
    }
}
