#pragma GCC optimize(2,"Ofast","inline","unroll-loops")
#include <bits/stdc++.h>
using namespace std;
#define rep(i,a,n) for (int i=a;i<=n;i++)
#define per(i,a,n) for (int i=a;i>=n;i--)
#define pb push_back
#define eb emplace_back
#define mp make_pair
#define all(x) (x).begin(),(x).end()
#define bit(x) (1ll<<(x))
#define fi first
#define se second
#define SZ(x) ((int)(x).size())
typedef vector<int> VI;
typedef long long ll;
typedef pair<int, int> PII;
typedef double db;
mt19937_64 rng(random_device {}());
typedef long double ldb;
typedef unsigned long long ull;
ll powmod(ll a,ll b,const ll p) { ll res=1; while (b) { if (b&1) res=res*a%p; b>>=1; a=a*a%p; } return res; }
// head

#ifdef DEBUG
#include "debug.cpp"
#else
#define debug(...) 42
#endif

const int mod = 1e9 + 7;
const ll inf = 1ll << 55;
const double pi = acosl(-1);
const double eps = 1e-12;
const int maxn = 2e5 + 105;
const int N = 2000005;

void solve() {}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int tt = 1;
    // cin >> tt;
    while (tt--) {
        solve();
    }
}