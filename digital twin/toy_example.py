import math
from random import choices
from scipy.stats import norm, truncnorm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


class ContextVariable:
    def __init__(self, name, values, distribution=None):
        self._name = name
        self._values = values
        self._distribution = distribution
        if type(distribution) is dict:
            self._distribution = [distribution[k] for k in values]

    def name(self):
        return '__CONTEXT__VARIABLE__' + self._name

    def values(self):
        return self._values

    def sample(self, nr=1):
        if self._distribution:
            return choices(self._values, weights=self._distribution, k=nr)
        else:
            return choices(self._values, k=nr)


CV_weekday = ContextVariable('weekday', range(7))
CV_weather = ContextVariable('weather', ['molto bassa', 'bassa', 'media', 'alta'],
                             {'media': 0.075, 'molto bassa': 0.65, 'bassa': 0.2, 'alta': 0.075})


class PresenceVariable:
    def __init__(self, cvs: list[ContextVariable], stats):
        self.cvs = cvs
        self.stats = stats

    def sample(self, cvs=None, nr=1):
        all_cvs = []
        if cvs is not None:
            for cv in self.cvs:
                # TODO: simplify!
                if cv.name() in cvs.keys():
                    all_cvs.append(cvs[cv.name()])
                else:
                    all_cvs.append(None)
        stats = self.stats(*all_cvs)
        return truncnorm.rvs(-stats['mean'] / stats['std'], 10, loc=stats['mean'], scale=stats['std'], size=nr)


def tourist_presences_stats(w=None, p=None):
    stats_map = {(-1, '*'): {'mean': 3292.75, 'std': 1127.9532044010873},
                 (0, '*'): {'mean': 3037.4615384615386, 'std': 1171.188258094076},
                 (1, '*'): {'mean': 2945.230769230769, 'std': 1117.096247259396},
                 (2, '*'): {'mean': 3031.4615384615386, 'std': 1151.6471258871368},
                 (3, '*'): {'mean': 3136.4615384615386, 'std': 1167.2054671582473},
                 (4, '*'): {'mean': 3361.230769230769, 'std': 994.882920569564},
                 (5, '*'): {'mean': 3858.5, 'std': 1151.389511717172},
                 (6, '*'): {'mean': 3635.3846153846152, 'std': 1051.836927986902},
                 (-1, 'media'): {'mean': 3374.0, 'std': 814.3095234614416},
                 (0, 'media'): {'mean': 3205.7307692307695, 'std': 976.5806430255634},
                 (1, 'media'): {'mean': 3159.6153846153848, 'std': 953.7620839425122},
                 (2, 'media'): {'mean': 3202.7307692307695, 'std': 968.3993093124825},
                 (3, 'media'): {'mean': 2884.0, 'std': 974.9187287888266},
                 (4, 'media'): {'mean': 3367.6153846153848, 'std': 900.0792392611492},
                 (5, 'media'): {'mean': 3619.0, 'std': 982.878425849301},
                 (6, 'media'): {'mean': 3504.6923076923076, 'std': 925.4841044492125},
                 (-1, 'molto bassa'): {'mean': 3897.4615384615386, 'std': 1067.3545326935837},
                 (0, 'molto bassa'): {'mean': 4197.2, 'std': 531.3583536559861},
                 (1, 'molto bassa'): {'mean': 2849.0, 'std': 1111.0871552973092},
                 (2, 'molto bassa'): {'mean': 1768.0, 'std': 1108.7000405335853},
                 (3, 'molto bassa'): {'mean': 4418.0, 'std': 788.4484764396466},
                 (4, 'molto bassa'): {'mean': 4071.2, 'std': 765.3529904560378},
                 (5, 'molto bassa'): {'mean': 4236.6, 'std': 1349.955295556116},
                 (6, 'molto bassa'): {'mean': 4130.333333333333, 'std': 919.204728737474},
                 (-1, 'bassa'): {'mean': 3494.875, 'std': 1279.9593115742837},
                 (0, 'bassa'): {'mean': 3266.1682692307695, 'std': 1224.3664960108872},
                 (1, 'bassa'): {'mean': 3220.0528846153848, 'std': 1195.758229578351},
                 (2, 'bassa'): {'mean': 3111.0, 'std': 620.8397538817887},
                 (3, 'bassa'): {'mean': 1653.0, 'std': 1222.2829075993866},
                 (4, 'bassa'): {'mean': 1827.0, 'std': 1128.4545441041175},
                 (5, 'bassa'): {'mean': 4564.666666666667, 'std': 140.25809542886762},
                 (6, 'bassa'): {'mean': 4563.0, 'std': 1160.3053349159973},
                 (-1, 'alta'): {'mean': 2839.3333333333335, 'std': 401.7229559451804},
                 (0, 'alta'): {'mean': 2670.0, 'std': 685.9250753616163},
                 (1, 'alta'): {'mean': 3298.0, 'std': 669.8979075383895},
                 (2, 'alta'): {'mean': 2935.397435897436, 'std': 680.1787174097348},
                 (3, 'alta'): {'mean': 2987.897435897436, 'std': 684.7577896323534},
                 (4, 'alta'): {'mean': 2550.0, 'std': 632.1924609409538},
                 (5, 'alta'): {'mean': 3348.916666666667, 'std': 680.1026379093822},
                 (6, 'alta'): {'mean': 3237.3589743589746, 'std': 650.0361835184224}}
    if w is None:
        w = -1
    if p is None:
        p = '*'
    return stats_map[w, p]


def excursionist_presences_stats(w=None, p=None):
    stats_map = {(-1, '*'): {'mean': 3198.0760869565215, 'std': 1849.4465714155635},
                 (0, '*'): {'mean': 2830.0, 'std': 2083.9047483030504},
                 (1, '*'): {'mean': 2861.0, 'std': 2259.344189508687},
                 (2, '*'): {'mean': 2746.230769230769, 'std': 1543.493070594863},
                 (3, '*'): {'mean': 2810.3076923076924, 'std': 1691.1208011560157},
                 (4, '*'): {'mean': 2940.153846153846, 'std': 1752.114714954182},
                 (5, '*'): {'mean': 3690.5, 'std': 1471.7294771275576},
                 (6, '*'): {'mean': 4470.461538461538, 'std': 1752.059388994592},
                 (-1, 'media'): {'mean': 2647.0, 'std': 1876.980820360187},
                 (0, 'media'): {'mean': 2738.5, 'std': 1977.7384164803868},
                 (1, 'media'): {'mean': 2754.0, 'std': 2059.307094704439},
                 (2, 'media'): {'mean': 2696.6153846153848, 'std': 1702.0889782456761},
                 (3, 'media'): {'mean': 1223.0, 'std': 1781.628835835903},
                 (4, 'media'): {'mean': 2793.576923076923, 'std': 1813.4733841553495},
                 (5, 'media'): {'mean': 3359.0, 'std': 2001.1121907579295},
                 (6, 'media'): {'mean': 3558.730769230769, 'std': 1813.4447521981024},
                 (-1, 'molto bassa'): {'mean': 4346.038461538462, 'std': 1769.3315682656935},
                 (0, 'molto bassa'): {'mean': 4691.8, 'std': 1950.4303884014932},
                 (1, 'molto bassa'): {'mean': 2945.5, 'std': 1906.8120865290668},
                 (2, 'molto bassa'): {'mean': 1126.0, 'std': 1652.5589294191113},
                 (3, 'molto bassa'): {'mean': 4809.333333333333, 'std': 1678.8878263104218},
                 (4, 'molto bassa'): {'mean': 4475.6, 'std': 1484.8472311992234},
                 (5, 'molto bassa'): {'mean': 4684.0, 'std': 1469.9692173647718},
                 (6, 'molto bassa'): {'mean': 5468.0, 'std': 1604.9012430676225},
                 (-1, 'bassa'): {'mean': 2776.875, 'std': 1316.2643500777929},
                 (0, 'bassa'): {'mean': 2803.4375, 'std': 1656.19127190948},
                 (1, 'bassa'): {'mean': 2818.9375, 'std': 1724.4982491164465},
                 (2, 'bassa'): {'mean': 2672.5, 'std': 287.79245994292484},
                 (3, 'bassa'): {'mean': 1114.0, 'std': 1491.9658247549303},
                 (4, 'bassa'): {'mean': 1049.0, 'std': 1518.6329828305797},
                 (5, 'bassa'): {'mean': 3376.0, 'std': 986.9240092327271},
                 (6, 'bassa'): {'mean': 4579.0, 'std': 1518.6090059500705},
                 (-1, 'alta'): {'mean': 1169.0, 'std': 697.5507150021423},
                 (0, 'alta'): {'mean': 900.0, 'std': 1205.6654789680063},
                 (1, 'alta'): {'mean': 1961.0, 'std': 1255.3912357618722},
                 (2, 'alta'): {'mean': 1957.6153846153845, 'std': 1037.624544329161},
                 (3, 'alta'): {'mean': 1989.6538461538462, 'std': 1086.1134949909122},
                 (4, 'alta'): {'mean': 646.0, 'std': 1105.5265135590664},
                 (5, 'alta'): {'mean': 2429.75, 'std': 1013.2156478559028},
                 (6, 'alta'): {'mean': 2819.730769230769, 'std': 1105.5090589947213}}
    if w is None:
        w = -1
    if p is None:
        p = '*'
    return stats_map[w, p]


PV_tourists = PresenceVariable([CV_weekday, CV_weather], tourist_presences_stats)
PV_excursionists = PresenceVariable([CV_weekday, CV_weather], excursionist_presences_stats)


class Constraint:
    def __init__(self, a, b, sigma):
        self.a = a
        self.b = b
        self.sigma = sigma

    def plot_line(self, x0, y0, x1, y1, delta):
        # @TODO: manage cases where x0 > x1 and/or y0 > y1
        delta_x = delta / (math.sqrt(1 + self.b ** 2))
        delta_y = delta_x * self.b
        r = math.ceil(abs((x1 - x0) / delta_x))
        xs = [x0 + delta_x * i for i in range(r) if y0 <= self.a + self.b * x0 + delta_y * i <= y1]
        ys = [self.a + self.b * x0 + delta_y * i for i in range(r) if y0 <= self.a + self.b * x0 + delta_y * i <= y1]
        return xs, ys

    def probability(self, x, y):
        dist = (self.b * x - y + self.a) / np.sqrt((self.b ** 2 + 1))
        prob = norm.cdf(dist, scale=self.sigma)
        return prob


c1 = Constraint(5000, -0.4, 800)
c2 = Constraint(16000, -3.0, 300)
c3 = Constraint(12000, -3.0, 300)

[x1, y1] = c1.plot_line(0, 0, 10000, 10000, 10)
[x2, y2] = c2.plot_line(0, 0, 10000, 10000, 10)
[x3, y3] = c3.plot_line(0, 0, 10000, 10000, 10)

xx = np.linspace(0, 10000, 100)
yy = np.linspace(0, 10000, 100)
xx, yy = np.meshgrid(xx, yy)
zz1 = c1.probability(xx, yy)
zz2 = c2.probability(xx, yy)
zz3 = c3.probability(xx, yy)

zzA = zz1 * zz2
zzB = zz1 * zz3

sample_tourists_all = PV_tourists.sample(nr=100)
sample_excursionists_all = PV_excursionists.sample(nr=100)

bad = {CV_weather.name(): 'alta'}
sample_tourists_bad = PV_tourists.sample(cvs=bad, nr=100)
sample_excursionists_bad = PV_excursionists.sample(cvs=bad, nr=100)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 5))
axA.plot(x1, y1, color='red')
axA.plot(x2, y2, color='red')
axA.contourf(xx, yy, zzA, levels=20)
axA.scatter(sample_excursionists_all, sample_tourists_all)
axA.set_title(f'Area = {zzA.sum():.2f}')
axB.plot(x1, y1, color='red')
axB.plot(x3, y3, color='red')
axB.contourf(xx, yy, zzB, levels=20)
axB.scatter(sample_excursionists_bad, sample_tourists_bad)
axB.set_title(f'Area = {zzB.sum():.2f}')
fig.colorbar(mappable=ScalarMappable(Normalize(0, 1)), ax=[axA, axB])
fig.show()
