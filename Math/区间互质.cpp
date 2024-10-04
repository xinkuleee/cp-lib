int p[N / 5], num;
void prime(int n) {
    num = 0;
    for (int i = 2; i * i <= n; i++) {
        if ((n % i) == 0) {
            p[++num] = i;
            while ((n % i) == 0) n /= i;
        }
    }
    if (n > 1) p[++num] = n;
}
ll solve(ll r, int k) {
    prime(k);
    ll res = 0;
    for (int i = 1; i < (1 << num); i++) {
        int k = 0;
        ll div = 1;
        for (int j = 1; j <= num; j++) {
            if (i & (1 << (j - 1))) {
                k++;
                div *= p[j];
            }
        }
        if (k % 2)
            res += r / div;
        else
            res -= r / div;
    }
    return r - res;
}
ll que(ll L, ll R, ll k) {
    return solve(R, k) - solve(L - 1, k);
}
