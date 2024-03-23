void gauss(int n) {
    int ans = 1;
    //rep(i,1,n) rep(j,1,n) p[i][j]%=mod;
    for (int i = 1; i <= n; i++) {
        for (int j = i + 1; j <= n; j++) {
            int x = i, y = j;
            while (p[x][i]) {
                int t = p[y][i] / p[x][i];
                for (int k = i; k <= n; k++)
                    p[y][k] = (p[y][k] - p[x][k] * t) % mod;
                swap(x, y);
            }
            if (x == i) {
                for (int k = i; k <= n; k++) swap(p[i][k], p[j][k]);
                ans = -ans;
            }
        }
    }
}
