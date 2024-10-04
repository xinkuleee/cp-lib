ll comb[N][N];
ll s[maxn], inv[maxn], p;
// 1^k+2^k+...+n^k
void solve() {
    cin >> k >> n >> p;
    rep(i, 0, k + 1) {
        comb[i][0] = comb[i][i] = 1;
        rep(j, 1, i - 1) {
            comb[i][j] = (comb[i - 1][j - 1] + comb[i - 1][j]) % p;
        }
    }
    inv[1] = 1;
    rep(i, 2, k + 1) inv[i] = (p - p / i) * inv[p % i] % p;
    assert(inv[k] * k % p == 1);

    ll pw = 1;
    // (k+1)*S[k]=(n+1)^(k+1)-[0-k-1](k+1,j)*S[j]-1
    rep(i, 0, k) {
        pw = pw * (n + 1) % p;
        s[i] = (pw - 1 + p) % p;
        rep(j, 0, i - 1) {
            s[i] = (s[i] - comb[i + 1][j] * s[j] % p + p) % p;
        }
        s[i] = s[i] * inv[i + 1] % p;
    }
    cout << s[k] << '\n';
}