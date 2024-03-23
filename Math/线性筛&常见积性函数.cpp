uint p[maxn], pe[maxn], prime[maxn]; 
// 因子个数  因子和  欧拉函数  莫比乌斯函数
uint d[maxn], f[maxn], phip[maxn], u[maxn];
uint tot;
void solve() {
    p[1] = 1;
    for (uint i = 2; i <= n; i++) {
        if (!p[i]) p[i] = i, pe[i] = i, prime[++tot] = i;
        for (uint j = 1; j <= tot && prime[j]*i <= n; j++) {
            p[prime[j]*i] = prime[j];
            if (prime[j] == p[i]) {
                pe[prime[j]*i] = pe[i] * p[i];
                break;
            } else {
                pe[prime[j]*i] = prime[j];
            }
        }
    }

    d[1] = 1;
    for (uint i = 2; i <= n; i++) {
        if (i == pe[i])
            d[i] = d[i / p[i]] + 1;
        else
            d[i] = d[i / pe[i]] * d[pe[i]];
    }

    f[1] = 1;
    for (uint i = 2; i <= n; i++) {
        if (i == pe[i])
            f[i] = f[i / p[i]] + i;
        else
            f[i] = f[i / pe[i]] * f[pe[i]];
    }

    phip[1] = 1;
    for (uint i = 2; i <= n; i++) {
        if (i == pe[i])
            phip[i] = i / p[i] * (p[i] - 1);
        else
            phip[i] = phip[i / pe[i]] * phip[pe[i]];
    }

    u[1] = (uint)1;
    for (uint i = 2; i <= n; i++) {
        if (i == pe[i])
            if (i == p[i]) u[i] = (uint) - 1;
            else u[i] = (uint)0;
        else
            u[i] = u[i / pe[i]] * u[pe[i]];
    }
}
