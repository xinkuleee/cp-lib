int bsgs(int a, int b, int m) { // a^x=b(mod m)
    int res = m + 1;
    int t = sqrt(m) + 2;
    ll d = powmod(a, t, m);
    ll cnt = 1;
    //map<int,int> p;
    hs.init();
    for (int i = 1; i <= t; i++) {
        cnt = cnt * d % m;
        //if (!p.count(cnt)) p[cnt] = i;
        if (hs.query(cnt) == -1) hs.insert(cnt, i);
    }
    cnt = b;
    for (int i = 1; i <= t; i++) {
        cnt = cnt * a % m;
        //if (p.count(cnt)) res = min(res, p[cnt] * t - i);
        int tmp = hs.query(cnt);
        if (tmp != -1) res = min(res, tmp * t - i);
    }
    if (res >= m) res = -1;
    return res;
}
