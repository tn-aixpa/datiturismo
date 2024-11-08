import math
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


class Constraint:
    def __init__(self, a, b, sigma):
        self.a = a
        self.b = b
        self.sigma = sigma

    def plot_line(self, x0, y0, x1, y1, delta):
        # @TODO: manage cases where x0 > x1 and/or y0 > y1
        delta_x = delta/(math.sqrt(1 + self.b**2))
        delta_y = delta_x*self.b
        r = math.ceil(abs((x1-x0)/delta_x))
        xs = [ x0 + delta_x*i for i in range(r) if y0 <= self.a + self.b*x0 + delta_y*i <= y1 ]
        ys = [ self.a + self.b*x0 + delta_y*i for i in range(r) if y0 <= self.a + self.b*x0 + delta_y*i <= y1 ]
        return xs,ys

    def probability(self, x, y):
        dist = (self.b*x - y + self.a) / np.sqrt((self.b**2 + 1))
        prob = norm.cdf(dist, scale=self.sigma)
        return prob


c1 = Constraint(5000, -0.4, 800)
c2 = Constraint(16000, -3.0, 300)
c3 = Constraint(12000, -3.0, 300)

[x1,y1] = c1.plot_line(0, 0, 10000, 10000, 10)
[x2,y2] = c2.plot_line(0, 0, 10000, 10000, 10)
[x3,y3] = c3.plot_line(0, 0, 10000, 10000, 10)

xx = np.linspace(0, 10000, 100)
yy = np.linspace(0, 10000, 100)
xx, yy = np.meshgrid(xx, yy)
zz1 = c1.probability(xx, yy)
zz2 = c2.probability(xx, yy)
zz3 = c3.probability(xx, yy)

zzA = zz1*zz2
zzB = zz1*zz3

fig, (axA,axB) = plt.subplots(1,2, figsize=(11,5))
axA.plot(x1,y1,color='red')
axA.plot(x2,y2,color='red')
axA.contourf(xx, yy, zzA, levels=20)
axA.set_title(f'Area = {zzA.sum():.2f}')
axB.plot(x1,y1,color='red')
axB.plot(x3,y3,color='red')
axB.contourf(xx, yy, zzB, levels=20)
axB.set_title(f'Area = {zzB.sum():.2f}')
fig.colorbar(mappable=ScalarMappable(Normalize(0,1)), ax=[axA,axB])
fig.show()
