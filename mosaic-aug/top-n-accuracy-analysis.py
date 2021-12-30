import numpy as np
import matplotlib.pyplot as plt


def calculate_threshold(a, n, k, t):
    """
    :param a:
    :param n:
    :param k:
    :param t:
    :return: threshold; return -1 if the result is nan
    """
    try:
        assert 0 < a < 1
        assert n <= k
        assert a * n <= 1.
        assert a >= (1 - n * a) / (k - n)
        assert a > 0
        assert (1 - n * a) / (k - n) > 0
        assert (k - n) > 0
        ln_a = np.log(a)
        ln_b = np.log((1 - n * a) / (k - n))
        assert (ln_b - ln_a) != 0
        thresh = (t + ln_b) / (ln_b - ln_a)
        return thresh
    # except AssertionError:
    except AssertionError:
        return -1


# Top-n accuracy analysis
def top_n_acc_threshold_by_uniform(start, distance, times):
    """
    Let b = (1-na)/(k-n)
    - p*ln(a) - (1-p)*ln(b) <= t
    :param start:
    :param distance:
    :param times:
    :var n: calculate based on top-n accuracy (same as n)
    :var k: total number of classes = 8
    :var a: prob applied to top-n (try each candidate)
    :var t: target maximum value = 1.65163
    /////////:return: How large top-n accuracy we should ensure to
    """
    k = 8
    t = 1.65163
    #
    for n in range(1, k):
        all_a = []
        all_thresh = []
        min_thresh = 999
        min_thresh_a = 999
        ranges = np.arange(float(times)) * distance + start
        for a in ranges:
            thresh = calculate_threshold(a, n, k, t)
            if 0 <= thresh <= 1:
                all_a.append(a)
                all_thresh.append(thresh)
                if thresh <= min_thresh:
                    min_thresh_a = a
                    min_thresh = thresh
        if min_thresh <= 1:
            print(f'When n = {n}, min thresh {min_thresh:.6f} occurs when a = {min_thresh_a:.6f}')
            plt.plot(all_a, all_thresh, label=f'n = {n}')
    plt.xlabel('Top-n accuracy threshold when assigning what prob for top-n classes')
    plt.ylabel('Threshold')
    plt.legend()
    plt.show()


top_n_acc_threshold_by_uniform(start=0.001,
                               distance=0.001,
                               times=1000)

