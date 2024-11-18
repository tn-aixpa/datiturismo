import math
import random
import numbers
from scipy import stats
import pandas as pd
import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


# CONTEXT VARIABLES

class ContextVariable(Symbol):
    def __new__(cls, name, *args, **kwargs):
        obj = Symbol.__new__(cls, name)
        return obj

    def __init__(self, name, values, distribution=None):
        self._values = values
        self._distribution = distribution

    def sample(self, nr=1, *, subset=None):
        values = self._values if subset is None else subset
        if self._distribution:
            distribution = [self._distribution[v] for v in values]
            return random.choices(values, weights=distribution, k=nr)
        else:
            return random.choices(values, k=nr)


# PRESENCE VARIABLES

class PresenceVariable(Symbol):
    def __new__(cls, name, *args, **kwargs):
        obj = Symbol.__new__(cls, name)
        return obj

    def __init__(self, name, cvs: list[ContextVariable], distribution):
        self._cvs = cvs
        self._distribution = distribution

    def sample(self, cvs=None, nr=1):
        all_cvs = []
        if cvs is not None:
            all_cvs = [cvs[cv] for cv in self._cvs if cv in cvs.keys()]
            # TODO: solve this issue of symbols vs names
            all_cvs = list(map(lambda v: v.name if isinstance(v, Symbol) else v, all_cvs))
        distr = self._distribution(*all_cvs)
        return stats.truncnorm.rvs(-distr['mean'] / distr['std'], 10, loc=distr['mean'], scale=distr['std'], size=nr)


# INDICES

class Index(Symbol):
    def __new__(cls, name, *args, **kwargs):
        obj = Symbol.__new__(cls, name)
        return obj

    def __init__(self, name, value, *, cvs=None):
        self._csv = cvs
        if cvs is not None:
            self._value = lambdify(cvs, value, 'numpy')
        else:
            self._value = value

    @property
    def csv(self):
        return self._csv

    @property
    def value(self):
        return self._value


# CONSTRAINTS

class Constraint:
    def __init__(self, *, usage, capacity):
        self._usage = usage
        self._capacity = capacity

    @property
    def usage(self):
        return self._usage

    @property
    def capacity(self):
        return self._capacity


# MODELS

class Model:
    def __init__(self, name,
                 cvs: list[ContextVariable], pvs: list[PresenceVariable],
                 indexes: list[Index], capacities: list[Index], constraints: list[Constraint]):
        self._name = name
        self._cvs = cvs
        self._pvs = pvs
        self._indexes = indexes
        self._capacities = capacities
        self._constraints = constraints

    def evaluate(self, p_case, c_case):
        c_df = pd.DataFrame(c_case)
        c_subs = {}
        for index in self._indexes:
            if index.csv is None:
                c_subs[index] = [index.value] * c_df.shape[0]
            else:
                args = [c_df[cv].values for cv in index.csv]
                c_subs[index] = index.value(args)
        probability = 1
        for constraint in self._constraints:
            usage = (lambdify(self._pvs + self._indexes, constraint.usage, 'numpy')
                     (*[np.expand_dims(p_case[pv], axis=(2,3)) for pv in self._pvs],
                      *[np.expand_dims(c_subs[index], axis=(0,1)) for index in self._indexes]))
            capacity = constraint.capacity
            # TODO: model type in declaration
            if isinstance(capacity.value, numbers.Number):
                result = (usage <= capacity.value)
            else:
                result = (1 - capacity.value.cdf(usage))
            probability *= result
        return probability.mean(axis=(2,3))


    # TODO: to be removed in the future
    def evaluate_single_case(self, p_case, c_case):
        c_subs = {}
        for index in self._indexes:
            if index.csv is None:
                c_subs[index] = index.value
            else:
                args = [c_case[cv] for cv in index.csv]
                c_subs[index] = index.value(*args)[()]
        probability = 1
        for constraint in self._constraints:
            usage = lambdify(self._pvs, constraint.usage.subs(c_subs), 'numpy')(*[p_case[pv] for pv in self._pvs])
            capacity = constraint.capacity
            # TODO: model type in declaration
            if isinstance(capacity.value, numbers.Number):
                result = (usage <= capacity.value)
            else:
                result = (1 - capacity.value.cdf(usage))
            probability *= result
        return probability

    @property
    def cvs(self):
        return self._cvs


# ENSEMBLE SIMULATIONS

class Ensemble:
    def __init__(self, model: Model, scenario, cv_ensemble_size=20):
        # TODO: what if cvs is empty?
        self._model = model
        self._ensemble = {}
        self._size = 1
        for cv in model.cvs:
            if cv in scenario.keys():
                if len(scenario[cv]) == 1:
                    self._ensemble[cv] = scenario[cv]
                else:
                    self._ensemble[cv] = cv.sample(cv_ensemble_size, subset=scenario[cv])
                    self._size *= cv_ensemble_size
            else:
                self._ensemble[cv] = cv.sample(cv_ensemble_size)
                self._size *= cv_ensemble_size

    def size(self):
        return self._size

    def __iter__(self):
        self._pos = {k: 0 for k in self._ensemble.keys()}
        self._pos[list(self._ensemble.keys())[0]] = -1
        return self

    def __next__(self):
        for k in self._ensemble.keys():
            self._pos[k] += 1
            if self._pos[k] < len(self._ensemble[k]):
                return {k: self._ensemble[k][self._pos[k]] for k in self._ensemble.keys()}
            self._pos[k] = 0
        raise StopIteration


# MODEL DEFINITION

# Context variables

CV_weekday = ContextVariable('weekday', range(7))
CV_weather = ContextVariable('weather', [Symbol(v) for v in ['molto bassa', 'bassa', 'media', 'alta']],
                             {Symbol('media'): 0.075, Symbol('molto bassa'): 0.65,
                              Symbol('bassa'): 0.2, Symbol('alta'): 0.075})

# Presence variables

from toy_aux import tourist_presences_stats, excursionist_presences_stats

PV_tourists = PresenceVariable('tourists', [CV_weekday, CV_weather], tourist_presences_stats)
PV_excursionists = PresenceVariable('excursionists', [CV_weekday, CV_weather], excursionist_presences_stats)

# Capacity indexes

I_C_parking = Index('parking capacity', 400)
I_C_parking_extended = Index('extended parking capacity', 600)
I_C_accommodation = Index('accommodation capacity', stats.lognorm(s=0.125, loc=0, scale=5000))

# Other indexes

I_U_tourists_parking = Index('tourist parking usage',
                             Piecewise((0.15, Eq(CV_weather, Symbol('alta'))), (0.20, True)),
                             cvs=[CV_weather])
I_U_excursionists_parking = Index('excursionist parking usage',
                                  Piecewise((0.55, Eq(CV_weather, Symbol('alta'))), (0.80, True)),
                                  cvs=[CV_weather])
I_U_tourists_accommodation = Index('tourist accommodation usage', 0.90)
I_Xa_tourists_per_vehicle = Index('tourists per vehicle', 2.5)
I_Xa_excursionists_per_vehicle = Index('excursionists per vehicle', 2.5)
I_Xa_tourists_per_accommodation = Index('tourists per accommodation', 1.05)
I_Xo_tourists_parking = Index('tourists rotation in parking', 3.0)
I_Xo_excursionists_parking = Index('excursionists rotation in parking', 3.0)

# Constraints

C_accommodation = Constraint(usage=PV_tourists * I_U_tourists_accommodation / I_Xa_tourists_per_accommodation,
                             capacity=I_C_accommodation)
C_parking = Constraint(usage=PV_tourists * I_U_tourists_parking / (I_Xa_tourists_per_vehicle * I_Xo_tourists_parking) +
                             PV_excursionists * I_U_excursionists_parking / (
                                         I_Xa_excursionists_per_vehicle * I_Xo_excursionists_parking),
                       capacity=I_C_parking)
# TODO: contraint should not be duplicated: original constraint should work...
C_parking_extended = Constraint(
    usage=PV_tourists * I_U_tourists_parking / (I_Xa_tourists_per_vehicle * I_Xo_tourists_parking) +
          PV_excursionists * I_U_excursionists_parking / (I_Xa_excursionists_per_vehicle * I_Xo_excursionists_parking),
    capacity=I_C_parking_extended)

# Models

# Base model
M_Base = Model('base model', [CV_weekday, CV_weather], [PV_tourists, PV_excursionists],
               [I_U_tourists_parking, I_U_excursionists_parking, I_U_tourists_accommodation,
                I_Xa_tourists_per_vehicle, I_Xa_excursionists_per_vehicle, I_Xa_tourists_per_accommodation,
                I_Xo_tourists_parking, I_Xo_excursionists_parking],
               [I_C_parking, I_C_accommodation], [C_accommodation, C_parking])

# Larger park capacity model
M_MoreParking = Model('larger parking model', [CV_weekday, CV_weather], [PV_tourists, PV_excursionists],
                      [I_U_tourists_parking, I_U_excursionists_parking, I_U_tourists_accommodation,
                       I_Xa_tourists_per_vehicle, I_Xa_excursionists_per_vehicle, I_Xa_tourists_per_accommodation,
                       I_Xo_tourists_parking, I_Xo_excursionists_parking],
                      [I_C_parking_extended, I_C_accommodation], [C_accommodation, C_parking_extended])

# ANALYSIS SITUATIONS

# Base situation
S_Base = {}

# Good weather situation
S_Good_Weather = {CV_weather: [Symbol('molto bassa'), Symbol('bassa')]}

# Bad weather situation
S_Bad_Weather = {CV_weather: [Symbol('alta')]}

# PLOTTING

(x_max, y_max) = (10000, 10000)
(x_sample, y_sample) = (100, 100)
target_presence_samples = 200
ensemble_size = 20  # TODO: make configurable; may it be a CV parameter?


def plot_scenario(ax, model, situation, title):
    ensemble = Ensemble(model, situation, cv_ensemble_size=ensemble_size)
    xx = np.linspace(0, x_max, x_sample + 1)
    yy = np.linspace(0, y_max, y_sample + 1)
    xx, yy = np.meshgrid(xx, yy)
    zz = model.evaluate({PV_tourists: xx, PV_excursionists: yy}, ensemble)
    #zz = sum([model.evaluate_single_case({PV_tourists: xx, PV_excursionists: yy}, case) for case in ensemble])/ensemble.size()

    case_number = ensemble.size()
    samples_per_case = math.ceil(target_presence_samples/case_number)

    sample_tourists = [sample for case in ensemble for sample in PV_tourists.sample(cvs=case, nr=samples_per_case)]
    sample_excursionists = [sample for case in ensemble for sample in PV_excursionists.sample(cvs=case, nr=samples_per_case)]

    if case_number*samples_per_case > target_presence_samples:
        sample_tourists = random.sample(sample_tourists, target_presence_samples)
        sample_excursionists = random.sample(sample_excursionists, target_presence_samples)

    # TODO: move elsewhere, it cannot be computed this way...
    area = zz.sum() * (x_max / x_sample / 1000) * (y_max / y_sample / 1000)

    # TODO: re-enable median
    # C_accommodation.median(cvs=scenario).plot(ax, color='red')
    # C_parking.median(cvs=scenario).plot(ax, color='red')
    ax.contourf(xx, yy, zz, levels=100, cmap='coolwarm_r')
    ax.scatter(sample_excursionists, sample_tourists, color='gainsboro', edgecolors='black')
    ax.set_title(f'{title}\nArea = {area:.2f} kp$^2$', fontsize=12)
    ax.set_xlim(left=0, right=x_max)
    ax.set_ylim(bottom=0, top=y_max)

import time
start_time = time.time()

fig, axs = plt.subplots(2, 3, figsize=(18, 10))
fig.subplots_adjust(hspace=0.3)
plot_scenario(axs[0, 0], M_Base, S_Base, 'Base')
plot_scenario(axs[0, 1], M_Base, S_Good_Weather, 'Good weather')
plot_scenario(axs[0, 2], M_Base, S_Bad_Weather, 'Bad weather')
plot_scenario(axs[1, 0], M_MoreParking, S_Base, 'More parking ')
plot_scenario(axs[1, 1], M_MoreParking, S_Good_Weather, 'More parking - Good weather')
plot_scenario(axs[1, 2], M_MoreParking, S_Bad_Weather, 'More parking - Bad weather')
fig.colorbar(mappable=ScalarMappable(Normalize(0, 1), cmap='coolwarm_r'), ax=axs)
fig.show()

print("--- %s seconds ---" % (time.time() - start_time))
