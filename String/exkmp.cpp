char s[maxn], a[maxn];
int z[maxn], p[maxn];

void exkmp(char s[], int len) {
    int L = 1, R = 0;
    z[1] = 0;
    rep(i, 2, len) {
        if (i > R) z[i] = 0;
        else {
            int k = i - L + 1;
            z[i] = min(z[k], R - i + 1);
        }
        while (i + z[i] <= len && s[z[i] + 1] == s[i + z[i]])
            z[i]++;
        if (i + z[i] - 1 > R)
            L = i, R = i + z[i] - 1;
    }
}

void match(char a[], char s[], int m, int n) {
    int L, R = 0;
    rep(i, 1, m) {
        if (i > R) p[i] = 0;
        else {
            int k = i - L + 1;
            p[i] = min(z[k], R - i + 1);
        }
        while (p[i] + 1 <= n && i + p[i] <= m && s[p[i] + 1] == a[p[i] + i])
            p[i]++;
        if (i + p[i] - 1 > R)
            L = i, R = i + p[i] - 1;
    }
}
