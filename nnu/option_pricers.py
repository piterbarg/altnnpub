import numpy as np
from scipy.stats import norm

# Implement down and out put option pricer and related utilities


def handle_edge_conditions(bs, ks, ts, vols):
    '''
    some basic handling of edge conditions
    '''
    bs = np.maximum(bs, 1e-3)
    ks = np.maximum(ks, 2e-3)
    ts = np.maximum(ts, 1e-2)
    vols = np.maximum(vols, 1e-4)

    return bs, ks, ts, vols


def black_scholes_bits(ks, ts, vols, rs):
    '''
    d1's, d2's, etc
    '''
    vts = vols*np.sqrt(ts)
    d1s = (np.log(1/ks) + (rs*ts + 0.5*vts*vts)) / vts
    d2s = d1s - vts
    ds = np.exp(-rs*ts)
    return vts, d1s, d2s, ds


def standard_put(ks, ts, vols, rs, handle_edge=True):
    '''
    Standard put BS normalized by spot. Mostly for testing our barrier formula
    All inputs are 1D numpy arrays
    '''
    # floor at small values
    if handle_edge:
        _, ks, ts, vols = handle_edge_conditions(0.0, ks, ts, vols)

    # standard BS
    _, d1s, d2s, ds = black_scholes_bits(ks, ts, vols, rs)

    Phi_p = norm.cdf(-d1s)
    Phi_m = norm.cdf(-d2s)

    return ds*ks*Phi_m - Phi_p


def reflected_put(bs, ks, ts, vols, rs, handle_edge=True):
    '''
    Needed in barrier pricing -- also normalized by spot
    All inputs are 1D numpy arrays
    '''
    bs2 = bs*bs
    return bs2*standard_put(ks/bs2, ts, vols, rs, handle_edge=handle_edge)


def down_out_put(bs, ks, ts, vols, rs):
    '''
    A down and out put in BS normalized by spot. All inputs are 1D numpy arrays
    from https://personal.ntu.edu.sg/nprivault/MA5182/barrier-options.pdf
    eq (11.12)
    bs: barriers over spots
    ks: strikes over spots
    ts: expiries
    vols: vols
    rs: discount rates
    '''

    # floor at small values
    bs, ks, ts, vols = handle_edge_conditions(bs, ks, ts, vols)

    # standard BS and reflected
    _, dk1s, dk2s, ds = black_scholes_bits(ks, ts, vols, rs)
    _, db1s, db2s, _ = black_scholes_bits(bs, ts, vols, rs)
    _, rk1s, rk2s, _ = black_scholes_bits(ks/(bs*bs), ts, vols, rs)
    _, rb1s, rb2s, _ = black_scholes_bits(1/bs, ts, vols, rs)
    rovs = 2*rs/(vols*vols)

    # not knocked out yet
    nkos = (bs <= 1.0)

    # strike less that barrier -- zero value
    # nzs indicates it is nonzero
    nzs = (ks > bs)

    # various terms
    Phi1p = norm.cdf(dk1s)
    Phi2p = norm.cdf(db1s)
    Phi3p = norm.cdf(rk1s)
    Phi4p = norm.cdf(rb1s)

    Phi1m = norm.cdf(dk2s)
    Phi2m = norm.cdf(db2s)
    Phi3m = norm.cdf(rk2s)
    Phi4m = norm.cdf(rb2s)

    term1 = (Phi1p - Phi2p) - np.power(bs, 1+rovs)*(Phi3p - Phi4p)
    term2 = ds * ks * \
        ((Phi1m - Phi2m) - np.power(1/bs, 1-rovs)*(Phi3m - Phi4m))

    return nzs * nkos * (term1 - term2)


def down_out_put_2(bs, ks, ts, vols, rs):
    '''
    A down and out put in BS normalized by spot. All inputs are 1D numpy arrays
    from https://people.maths.ox.ac.uk/howison/barriers.pdf
    bs: barriers over spots
    ks: strikes over spots
    ts: expiries
    vols: vols
    rs: discount rates

    This is for testing only -- likely slower than the main
    '''

    # floor at small values
    bs, ks, ts, vols = handle_edge_conditions(bs, ks, ts, vols)

    # not knocked out yet
    nkos = 1.0*(bs <= 1.0)

    # strike less that barrier -- zero value
    # nzs indicates it is nonzero
    nzs = 1.0*(ks > bs)

    # need d1 for strike = Barrier
    _, _, d2s, ds = black_scholes_bits(bs, ts, vols, rs)
    rovs = 2*rs/(vols*vols)

    putKp = standard_put(ks, ts, vols, rs, handle_edge=False)
    putBp = standard_put(bs, ts, vols, rs, handle_edge=False)
    digip = ds*norm.cdf(-d2s)

    # reflected puts
    putKm = reflected_put(bs, ks, ts, vols, rs, handle_edge=True)
    putBm = reflected_put(bs, bs, ts, vols, rs, handle_edge=True)
    digim = ds - digip

    term1 = putKp - putBp - (ks-bs)*digip
    term2 = putKm - putBm + (ks-bs)*digim

    # plt.figure()
    # plt.plot(ks, term1, label='t1')
    # plt.plot(ks, term2, label='t2')
    # plt.legend(loc='best')
    # plt.show()

#    return term1 - np.power(bs, 1-rovs) # this what _2 function returns it seems
    return nzs * nkos * (term1 - np.power(1/bs, 1-rovs) * term2)
