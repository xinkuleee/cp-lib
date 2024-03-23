#include <bits/extc++.h>
using namespace std;
#define rep(i,a,b) for (int i=a;i<=b;i++)
#define per(i,b,a) for (int i=b;i>=a;i--)
typedef long long ll;
typedef unsigned long long ull;
typedef double db;
typedef long double ldb;
typedef pair<int, int> PII;
typedef pair<long long, long long> PLL;
typedef pair<double, double> PDD;
typedef vector<int> VI;
typedef vector<long long> VLL;
mt19937_64 rng(random_device {}());
template<typename ...T>
void debug_out(T... args) {((cerr << args << " "), ...); cerr << '\n';}
#define pb push_back
#define eb emplace_back
#define fi first
#define se second
#define mp make_pair
#define bit(x) (1ll<<(x))
#define SZ(x) ((int)x.size())
#define all(x) x.begin(),x.end()
#define debug(...) cerr << "[" << #__VA_ARGS__ << "]:", debug_out(__VA_ARGS__)
ll powmod(ll a, ll b, const ll p) {ll res = 1; while (b) {if (b & 1) res = res * a % p; b >>= 1; a = a * a % p;} return res;}
