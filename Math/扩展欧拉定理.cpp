// mod [min(b, b % phi + phi)]
ll calc(ll p) {  
    if (p == 1) return 0;
    int phi = p, q = p;
    for (int i = 2; i * i <= p; i++) {
        if (q % i == 0) {
            phi = phi / i * (i - 1);
            while (q % i == 0) q /= i;
        }
    }
    if (q != 1) phi = phi / q * (q - 1);
    return powmod(2, calc(phi) + phi, p);
}
