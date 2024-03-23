int nxt[maxn];
char a[maxn], b[maxn];
vector<int> pos;
void get_next(char s[], int len) {
    nxt[1] = 0;
    int x = 0;
    for (int i = 2; i <= len; i++) {
        while (x > 0 && s[x + 1] != s[i]) x = nxt[x];
        if (s[x + 1] == s[i])
            nxt[i] = x + 1, x++;
        else nxt[i] = x;
    }
}
int match(char a[], char s[], int n, int m) {
    int x = 0, ans = 0;
    for (int i = 1; i <= n; i++) {
        while (x > 0 && a[i] != s[x + 1]) x = nxt[x];
        if (a[i] == s[x + 1]) x++;
        if (x == m) ans++, x = nxt[x], pos.pb(i - m + 1);
    }
    return ans;
}