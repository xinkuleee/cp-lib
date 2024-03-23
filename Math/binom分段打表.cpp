ll fact[] = {};

ll fct(ll x){
    ll res=fact[x/1000000];
    for(ll i=x/1000000*1000000+1;i<=x;i++){
        res=res*i%mod;
    }
    return res;
}

ll binom(ll a,ll b){
    if(b<0||b>a) return 0;
    return fct(a)*powmod(fct(b)*fct(a-b)%mod,mod-2,mod)%mod;
}
