import numpy as np
from scipy.optimize import minimize


class counter:

    def __init__(self,
                 c: np.ndarray[int, np.dtype[np.float64]],
                 alf: float = 0.5,
                 m: float = 1,
                 step=10**-5) -> None:
        self.c = c / m
        self.alf = alf
        self.step = step

    def costfun(self):
        return self.c**self.alf

    def p_from_l(self, l1: float, l2: float):
        return np.exp(-(self.c * l1 + l2) / self.costfun() - 1)

    def cnds(self, p: np.ndarray | list):
        p2 = np.array(p)
        cnd1 = np.sum(p2) - 1  # сумма вероятностей == 1
        cnd2 = np.sum(p2 * self.c) - 1  # матожидание == 1
        return np.array([cnd1, cnd2])

    def functional(self, cnd: np.ndarray | list):
        cnd2 = np.array(cnd)
        return sum(cnd2**2)

    def solve(self) -> tuple[float, float]:
        return minimize(
            lambda x: self.functional(self.cnds(self.p_from_l(*x))),
            x0=[1, 1],
            method='Nelder-Mead',
            tol=1e-10).x

    def _get_p_floor(self, p):
        return np.floor(p / self.step) * self.step

    def _get_dm(self, p_floor):
        s, dm = self.cnds(p_floor)
        return -dm

    def _get_n(self, p_floor):
        s, dm = self.cnds(p_floor)
        return round(-s / self.step)

    def _get_c_norm(self, dm, n):
        return self.c * self.step - dm / n

    def round_p(self, p):
        p_floor = self._get_p_floor(p)
        dm = self._get_dm(p_floor)
        n = self._get_n(p_floor)
        c_norm = self._get_c_norm(dm, n)
        sample = find_sample(c_norm, n)
        p_floor[sample] += self.step
        return p_floor


def gen_prices(n: int, d: float = 3, seed=None):
    if seed is not None:
        np.random.seed(seed)
    c = np.random.randn(n) * d
    c2 = np.exp(c)
    return c2 / c2.mean()


def find_sample(c, n):
    sample = abs(c).argsort()[:n]
    ost = abs(c).argsort()[n:]
    while True:
        # print(sample)
        # print(ost)
        # print('sum: {}'.format(c[sample].sum()))
        pairs = np.array([(i, j) for i in sample for j in ost])
        diff = c[pairs][:, 1] - c[pairs][:, 0]
        min_ind = abs(diff + c[sample].sum()).argmin()
        if abs(diff[min_ind] + c[sample].sum()) >= abs(c[sample].sum()):
            break
        else:
            sample = np.append(sample, pairs[min_ind, 1])
            del_ind = np.where(sample == pairs[min_ind, 0])
            sample = np.delete(sample, del_ind)
            ost = np.append(ost, pairs[min_ind, 0])
            del_ind = np.where(ost == pairs[min_ind, 1])
            ost = np.delete(ost, del_ind)
    return sample


def visualize(c, p, log_x=True, log_y=True):
    import pandas as pd
    import plotly.express as px

    data = pd.DataFrame({'P': p, 'cost': c})
    data_s = data.groupby(by='cost').agg([np.mean, np.size])
    data_s.columns = data_s.columns.get_level_values(1)
    data_s.reset_index(inplace=True)
    data_s.rename(columns={'mean': 'P'}, inplace=True)
    return px.scatter(data_s,
                      x='cost',
                      y='P',
                      size='size',
                      log_x=log_x,
                      log_y=log_y)
