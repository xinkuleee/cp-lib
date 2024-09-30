db rnd(db l, db r) {
  static uniform_real_distribution<db> u(0, 1);
  static default_random_engine e(rng());
  return l + (r - l) * u(e);  // u(rng);
}

db eval(pair<db, db> x) { ... }

void simulate_anneal() {
  pair<db, db> cur(rnd(0, 10000), rnd(0, 10000));
  for (double k = 10000; k > 1e-5; k *= 0.99) {
    // [start, end, step]
    pair<db, db> nxt(cur.fi + rnd(-k, k), cur.se + rnd(-k, k));
    db delta = eval(nxt) - eval(cur);
    if (exp(-delta / k) > rnd(0, 1)) {
      cur = nxt;
    }
  }
}

/**
 * https://codeforces.com/gym/104813/submission/234982955
 * The 9th CCPC (Harbin) 2023
 * Author: QwertyPi
 */
LD Prob() {
  static uniform_real_distribution<> dist(0.0, 1.0);
  return dist(rng);
}
LD Sigma(LD x) { return 1 / (1 + exp(-x)); }

LD overall_max_score = 0;
for (int main_loop = 0; main_loop < 5; main_loop++) {
  vector<LD> e(n, (LD)1 / n);
  for (int tr = 0; tr < 1000; tr++) {
    vector<LD> ne(n);
    for (int i = 0; i < n; i++) {
      ne[i] = Prob();
    }
    LD s = accumulate(all(ne), 0.0L);
    for (int i = 0; i < n; i++) {
      ne[i] /= s;
    }
    if (eval(ne) > eval(e)) e = ne;
  }
  LD t = (LD)0.0002;
  LD max_score = 0;
  const LD depr = 0.999995;
  const int tries = 2E6;
  const int loop = 1E5;

  LD score_old = eval(e);
  for (int tr = 0; tr < tries; tr++) {
#ifdef LOCAL
    if (tr % loop == loop - 1) {
      cout << fixed << setprecision(10) << "current score = " << max_score
           << ", t = " << t << '\n';
    }
#endif
    int x = rng() % n, y = rng() % n;
    if (e[x] < t || x == y) {
      t *= depr;
      continue;
    }
    e[x] -= t;
    e[y] += t;
    LD score_new = eval(e);
    if (score_new > score_old) {  // ok
      ;
    } else {  // revert
      e[x] += t;
      e[y] -= t;
    }
    score_old = score_new;
    max_score = max(max_score, score_new);
    t *= depr;
  }
  overall_max_score = max(overall_max_score, max_score);
#ifdef LOCAL
  cout << "Loop #" << main_loop << ": " << max_score << '\n';
#endif
}