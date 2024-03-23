pair<db,db> p[N];
db ans = 1e10;
db rd(db l, db r) {
    uniform_real_distribution<db> u(l,r);
    // uniform_int_distribution<ll> u(l,r);
    default_random_engine e(rng());
    return u(e);  // e(rng)
}

db dist(pair<db,db> a, pair<db,db> b) {
    db dx = a.fi - b.fi;
    db dy = a.se - b.se;
    // sqrtl() for long double
    return sqrt(dx * dx + dy * dy);
}

db eval(pair<db,db> x) {
    db res = 0;
    rep(i, 1, n) res += dist(p[i], x);
    ans = min(ans, res);
    return res;
}

void simulate_anneal() {
    pair<db,db> cnt(rd(0, 10000), rd(0, 10000));
    for (double k = 10000; k > 1e-5; k *= 0.99) {
        // [start, end, step]
        pair<db,db> np(cnt.fi + rd(-k, k), cnt.se + rd(-k, k));
        db delta = eval(np) - eval(cnt);
        if (exp(-delta / k) > rd(0, 1)) cnt = np;
    }
}
