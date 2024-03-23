uint pr[maxn], p[maxn], pe[maxn], u[maxn], tot;
uint g[maxn], f[maxn];
int main() {
    p[1] = 1;
    for (int i = 2; i <= n; i++) {
        if (!p[i]) pe[i] = i, p[i] = i, pr[++tot] = i;
        for (uint j = 1; j <= tot && pr[j]*i <= n; j++) {
            p[pr[j]*i] = pr[j];
            if (pr[j] == p[i]) {
                pe[pr[j]*i] = pe[i] * p[i];
                break;
            } else {
                pe[pr[j]*i] = pr[j];
            }
        }
    }
    u[1] = 1;
    for (uint i = 2; i <= n; i++) {
        if (i == pe[i]) {
            if (i == p[i]) u[i] = (uint) - 1;
            else u[i] = (uint)0;
        } else {
            u[i] = u[pe[i]] * u[i / pe[i]];
        }
    }
    for (int i = 1; i <= n; i++)
        for (int j = 1; i * j <= n; j++) {
            g[i * j] += f[i] * u[j];
        }
}