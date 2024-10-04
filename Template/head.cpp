#pragma GCC optimize("Ofast", "inline", "unroll-loops")
#include <bits/stdc++.h>
using namespace std;
#define rep(i, a, n) for (int i = a; i <= n; i++)
#define per(i, a, n) for (int i = a; i >= n; i--)
#define pb push_back
#define eb emplace_back
#define all(x) (x).begin(), (x).end()
#define bit(x) (1ll << (x))
#define fi first
#define se second
#define SZ(x) ((int)(x).size())
using VI = vector<int>;
using PII = pair<int, int>;
using ll = long long;
using ull = unsigned long long;
using db = double;
using ldb = long double;
mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
// head

#ifdef DEBUG
#include "debug.cpp"
#else
#define debug(...) 42
#endif

void solve() {}
int main() {
    cin.tie(nullptr)->sync_with_stdio(false);
    cout << fixed << setprecision(16);
    int tt = 1;
    cin >> tt;
    while (tt--) {
        solve();
    }
}