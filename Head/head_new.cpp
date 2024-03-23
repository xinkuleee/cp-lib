#include <bits/extc++.h>
using namespace std;
template<typename ...T>
void debug_out(T... args) {((cerr << args << " "), ...); cerr << '\n';}
#define debug(...) cerr<<"["<<#__VA_ARGS__<<"]:",debug_out(__VA_ARGS__)
#define rep(i,a,n) for (int i=a;i<n;i++)
#define per(i,a,n) for (int i=n-1;i>=a;i--)
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
ll powmod(ll a, ll b, const ll p) {ll res = 1; while (b) {if (b & 1) res = res * a % p; b >>= 1; a = a * a % p;} return res;}
// head