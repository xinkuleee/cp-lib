void fst(VI &a,bool inv) {
    for (int n=SZ(a),step=1;step<n;step*=2) {
        for (int i=0;i<n;i+=2*step) rep(j,i,i+step-1) {
            int &u=a[j],&v=a[j+step]; 
            tie(u,v)=
            inv?PII(v-u,u):PII(v,u+v); // AND
            inv?PII(v,u-v):PII(u+v,u); // OR
            PII(u+v,u-v); // XOR
        }
    }
    if (inv) for (auto &x : a) x/=SZ(a); // XOR only
}
VI conv(VI a,VI b) {
    fst(a,0),fst(b,0);
    rep(i,0,SZ(a)-1) a[i]=a[i]*b[i];
    fst(a,1); return a;
}
