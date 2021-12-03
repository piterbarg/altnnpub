import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

from nnu import option_pricers as op


def test_down_out_put_01(spot=100, strike=100, barrier=75, vol=0.2, expiry=5, npts=100, rate=0.2, plot_figures=False):
    spots = np.linspace(spot*0.5, spot*1.5, num=npts, endpoint=True)
    ks = strike/spots
    bs = np.ones_like(ks) * barrier/spots
    vols = np.ones_like(ks) * vol
    ts = np.ones_like(ks) * expiry
    rs = np.ones_like(ks) * rate

    ds = np.exp(-rs*ts)
    puts = op.standard_put(ks, ts, vols, rs)*spots
    rputs = op.reflected_put(bs, ks, ts, vols, rs)*spots
    kos = op.down_out_put(bs, ks, ts, vols, rs) * spots
    kos2 = op.down_out_put_2(bs, ks, ts, vols, rs) * spots

    if plot_figures:
        plt.figure()
        plt.plot(spots, np.maximum(ds*(ks-1/ds)*spots, 0.0), label='intr')
        plt.plot(spots, puts, label='std')
        plt.plot(spots, rputs, label='refl')
        plt.plot(spots, kos, label='down_out')
        plt.plot(spots, kos, label='down_out_2')
        plt.legend(loc='best')
        plt.show()


def test_down_out_put_vs_strike_01(spot=100, strike=100, barrier=75, vol=0.2, expiry=5, npts=100, rate=0.2, plot_figures=False):
    strikes = np.linspace(strike*0.5, strike*1.5, num=npts, endpoint=True)
    ks = strikes/spot

    bs = barrier/spot
    vols = vol
    ts = expiry
    rs = rate

    ds = np.exp(-rs*ts)
    puts = op.standard_put(ks, ts, vols, rs)*spot
    rputs = op.reflected_put(bs, ks, ts, vols, rs)*spot
    kos = op.down_out_put(bs, ks, ts, vols, rs) * spot
    kos2 = op.down_out_put(bs, ks, ts, vols, rs) * spot

    if plot_figures:
        plt.figure()
        plt.plot(strikes, np.maximum(ds*(ks-1/ds)*spot, 0.0), label='intr')
        plt.plot(strikes, puts, label='std')
#        plt.plot(strikes, rputs, label='refl')
        plt.plot(strikes, kos, label='down_out')
        plt.plot(strikes, kos2, label='down_out_2')
        plt.legend(loc='best')
        plt.show()


def test_down_out_put_2D_01(
        spot: float = 100,
        spot_range_rel=(0.5, 1.5),
        strike: float = 100,
        strike_range_rel=(0.5, 1.5),
        barrier: float = 50,
        barrier_range_rel=(0.5, 1.5),
        vol: float = 0.2,
        vol_range_rel=(0.25, 2.5),
        expiry=1,
        expiry_range_rel=(0.5, 2),
        rate=0.05,
        rate_range_rel=(0.0, 2),
        npts=101,
        plot_figures=False):

    spots = np.linspace(spot*spot_range_rel[0], spot *
                        spot_range_rel[1], num=npts, endpoint=True)
    strikes = np.linspace(strike*strike_range_rel[0], strike *
                          strike_range_rel[1], num=npts, endpoint=True)
    barriers = np.linspace(barrier*barrier_range_rel[0], barrier *
                           barrier_range_rel[1], num=npts, endpoint=True)
    expiries = np.linspace(expiry * expiry_range_rel[0], expiry *
                           expiry_range_rel[1], num=npts, endpoint=True)
    rates = np.linspace(rate * rate_range_rel[0], rate *
                        rate_range_rel[1], num=npts, endpoint=True)

    # fix strike and barrier
    spots_2d, expiries_2d = np.meshgrid(spots, expiries)
    spots_1d = spots_2d.reshape(-1)
    expiries_1d = expiries_2d.reshape(-1)

    ks = strike/spots_1d
    bs = barrier/spots_1d
    vols = np.ones_like(ks) * vol
    ts = expiries_1d
    rs = np.ones_like(ks) * rate

    kos = op.down_out_put(bs, ks, ts, vols, rs) * spots_1d

    if plot_figures:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(spots_1d, expiries_1d, kos,  # c=inputY,
                   cmap=cm.coolwarm, marker='.', label='y')

        plt.title('spot x expiry')
        plt.show()

    # fix spot and expiry
    strikes_2d, barriers_2d = np.meshgrid(strikes, barriers)
    strikes_1d = strikes_2d.reshape(-1)
    barriers_1d = barriers_2d.reshape(-1)

    ks = strikes_1d/spot
    bs = barriers_1d/spot
    vols = np.ones_like(ks) * vol
    ts = np.ones_like(ks) * expiry
    rs = np.ones_like(ks) * rate

    kos = op.down_out_put(bs, ks, ts, vols, rs) * spot

    if plot_figures:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(strikes_1d, barriers_1d, kos,  # c=inputY,
                   cmap=cm.coolwarm, marker='.', label='y')

        plt.title('strike x barrier')
        plt.show()
