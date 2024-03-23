ll w[N][N],sum[N][N],opt[N],dp[805][N];

ll calc(int i,int j) { return sum[j][j]-sum[j][i]-sum[i][j]+sum[i][i]; }

void rec(int d,int l,int r,int optl,int optr) {
    if (l>r) return;
    int md=(l+r)>>1;
    rep(i,optl,optr) if (dp[d-1][i]+calc(i,md)<dp[d][md]) {
        dp[d][md]=dp[d-1][i]+calc(i,md);
        opt[md]=i;
    }
    rec(d,l,md-1,optl,opt[md]);
    rec(d,md+1,r,opt[md],optr);
}