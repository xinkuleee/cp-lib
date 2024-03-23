#include<bits/stdc++.h>
using namespace std;
#define ll long long
#define db double
#define u64  unsigned long long
#define ldb long double
#define pb push_back
#define eb emplace_back
#define fi first
#define se second
#define bit(x) (1ll<<(x))
#define rep(i,a,b) for(int i=a;i<=b;i++)
#define per(i,b,a) for(int i=b;i>=a;i--)
#define ALL(x) x.begin(),x.end()
#define PII pair<int,int>
#define PLL pair<long long,long long>
#define VI vector<int>
#define VLL vector<long long>
#define debug cout<<" *** : "
std::mt19937 mrand(std::random_device{}());
ll gcd(ll a,ll b){return b==0?a:gcd(b,a%b);}
ll fpow(ll a,ll b){ll res=1;while(b){if(b&1) res=res*a;b>>=1;a=a*a;}return res;}
ll fpowmod(ll a,ll b,ll p){ll res=1;while(b){if(b&1) res=res*a%p;b>>=1;a=a*a%p;}return res;}


const ll mod = 998244353;
const int inf = 1<<30;
const ldb pi=acos(-1);
const double eps = 1e-10;
const int maxn = 1e5+105;
const int N = 505;


int T=1;
ll n,m,q;
int r;
struct linear_base{
    ll w[64];
    bool zero=0;
    ll tot=0;
    void clear(){
        rep(i,0,63) w[i]=0;
        zero=0;
        tot=0;
    }
    void insert(ll x){
        for(int i=62;i>=0;i--){
            if(x&bit(i))
                if(!w[i]){w[i]=x;return;}
                else x^=w[i];
        }
        zero=1;
    }
    void init(){
        for(int i=0;i<=62;i++){
            for(int j=0;j<i;j++){
                if(w[i]&bit(j))
                    w[i]^=w[j];
            }
        }
        for(int i=0;i<=62;i++)
            if(w[i]) w[tot++]=w[i];
    }
    ll qmax(){
        ll res=0;
        for(int i=62;i>=0;i--){
            res=max(res,res^w[i]);
        }
        return res;
    }
    bool check(ll x){
        for(int i=62;i>=0;i--){
            if(x&bit(i))
                if(!w[i]) return false;
                else x^=w[i];
        }
        return true;
    }
    ll query(ll k){
        ll res=0;
        k-=zero;
        if(k>=bit(tot)) return -1;
        for(int i=0;i<=62;i++){
            if(k&bit(i)) res^=w[i];
        }
        return res;
    }
}f;

void solve(){
    cin>>n;
    f.clear();
    rep(i,1,n){
        ll t;
        cin>>t;
        f.insert(t);
    }
    f.init();
    cin>>q;
    cout<<"Case #"<<r<<":"<<'\n';
    rep(i,1,q){
        ll k;
        cin>>k;
        cout<<f.query(k)<<'\n';
    }

}

int main(){
    std::ios::sync_with_stdio(false);
    cin.tie(0);cout.tie(0);
    cin>>T;
    while(T--){
        r++;
        solve();
    }
}