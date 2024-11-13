import math
import random
from scipy.stats import norm, truncnorm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


# CONTEXT VARIABLES

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
            return random.choices(self._values, weights=self._distribution, k=nr)
        else:
            return random.choices(self._values, k=nr)


# PRESENCE VARIABLES

class PresenceVariable:
    # TODO: replace stats with distribution?
    def __init__(self, cvs: list[ContextVariable], stats):
        self._cvs = cvs
        self._stats = stats

    def sample(self, cvs=None, nr=1):
        all_cvs = []
        if cvs is not None:
            for cv in self._cvs:
                # TODO: simplify!
                if cv in cvs.keys():
                    all_cvs.append(cvs[cv])
        stats = self._stats(*all_cvs)
        return truncnorm.rvs(-stats['mean'] / stats['std'], 10, loc=stats['mean'], scale=stats['std'], size=nr)


# PROBABILITY FIELDS

# TODO: not sure this is the right approach... SymPy? Other?
class GeoPlane:
    def __init__(self, params):
        self._params = params

    def plot(self, ax, **params):
        a = self._params[0]
        b = self._params[1]
        if a == 0:
            ax.axhline(y=1 / b, **params)
        elif b == 0:
            ax.axvline(x=1 / a, **params)
        else:
            ax.plot([0, 1 / a], [1 / b, 0], **params)


class ProbabilityField:
    # TODO: check if dims is of any use...
    def __init__(self, dims: int):
        self._dims = dims

    def dims(self):
        return self._dims

    def probability(self, coordinates: list[float], cvs: list) -> float:
        pass

    def median(self, cvs):
        pass


class PlanarGaussianProbabilityField(ProbabilityField):
    def __init__(self, dims, params, sigma):
        super().__init__(dims)
        self._params = params
        self._sigma = sigma

    def probability(self, coordinates, cvs):
        # TODO: optimize / manage np
        params = self._params(*cvs) if cvs is not None else self._params
        coordinates_array = np.array(coordinates)
        coordinates_array = np.rollaxis(coordinates_array, 0, coordinates_array.ndim)
        dist = (1 - np.dot(coordinates_array, params)) / math.sqrt(np.dot(params, params))
        prob = norm.cdf(dist, scale=self._sigma)
        return prob

    def median(self, cvs):
        params = self._params(*cvs) if cvs is not None else self._params
        return GeoPlane(params)


# CONSTRAINTS

class Constraint:
    def __init__(self, *, pvs: list[PresenceVariable], cvs=None, distribution: ProbabilityField):
        if cvs is None:
            cvs = []
        assert len(pvs) == distribution.dims()
        self._pvs = pvs
        self._cvs = cvs
        self._distribution = distribution

    def median(self, cvs=None):
        cvs_list = None
        if self._cvs:
            cvs_list = []
            for cv in self._cvs:
                cvs_list.append(cvs[cv])
        return self._distribution.median(cvs_list)

    def probability(self, pv_values: list[float], cvs: dict = None):
        cvs_list = None
        if self._cvs:
            cvs_list = []
            for cv in self._cvs:
                cvs_list.append(cvs[cv])
        return self._distribution.probability(pv_values, cvs_list)


# MODEL DEFINITION

# Context variables
CV_weekday = ContextVariable('weekday', range(7))
CV_weather = ContextVariable('weather', ['molto bassa', 'bassa', 'media', 'alta'],
                             {'media': 0.075, 'molto bassa': 0.65, 'bassa': 0.2, 'alta': 0.075})

# Presence variables
from toy_aux import tourist_presences_stats, excursionist_presences_stats
PV_tourists = PresenceVariable([CV_weekday, CV_weather], tourist_presences_stats)
PV_excursionists = PresenceVariable([CV_weekday, CV_weather], excursionist_presences_stats)

# Constraints
C_accommodation = Constraint(pvs=[PV_tourists, PV_excursionists],
                             distribution=PlanarGaussianProbabilityField(2, [1 / 5000, 0], sigma=400))
C_parking = Constraint(pvs=[PV_tourists, PV_excursionists],
                       cvs=[CV_weather],
                       distribution=PlanarGaussianProbabilityField(2,
                                                                   lambda w: [1 / 14000, 1 / 6000] if w != 'alta'
                                                                   else [1 / 10000, 1 / 4000],
                                                                   sigma=800))


# ENSEMBLE SIMULATION

# TODO: make configurable; may it be a CV parameter?
cv_ensemble_size = 20


class Ensemble:
    def __init__(self, cvs, scenario):
        # TODO: what if cvs is empty?
        self._ensemble = {}
        self._size = 1
        for cv in cvs:
            variants = cv.values()
            if cv in scenario.keys():
                variants = scenario[cv]
            if len(variants) == 1:
                self._ensemble[cv] = variants
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


# ANALYSIS SCENARIOS

# Base scenario

S_Base = {}

# Good weather

S_Good_Weather = { CV_weather: ['molto bassa'] }

# Bad weather

S_Bad_Weather = { CV_weather: ['alta'] }

# PLOTTING

(x_max, y_max) = (10000, 10000)

def plot_scenario(ax, scenario, title):
    ensemble = Ensemble([CV_weekday, CV_weather], scenario)
    xx = np.linspace(0, x_max, 100)
    yy = np.linspace(0, y_max, 100)
    xx, yy = np.meshgrid(xx, yy)
    zz_accommodation = sum([C_accommodation.probability([xx, yy], case) for case in ensemble])
    zz_parking = sum([C_parking.probability([xx, yy], case) for case in ensemble])
    zz = zz_accommodation * zz_parking

    target_samples = 200
    case_number = ensemble.size()
    samples_per_case = math.ceil(target_samples/case_number)

    sample_tourists = [ sample for case in ensemble for sample in PV_tourists.sample(cvs=case, nr=samples_per_case)]
    sample_excursionists = [ sample for case in ensemble for sample in PV_excursionists.sample(cvs=case, nr=samples_per_case)]

    if case_number*samples_per_case > target_samples:
        sample_tourists = random.sample(sample_tourists, target_samples)
        sample_excursionists = random.sample(sample_excursionists, target_samples)

    # TODO: move elsewhere, it cannot be computed this way...
    # TODO: fix the unit (square-persons)
    area = zz.sum()/(ensemble.size()**2)

    # TODO: re-enable median
    #C_accommodation.median(cvs=scenario).plot(ax, color='red')
    #C_parking.median(cvs=scenario).plot(ax, color='red')
    ax.contourf(xx, yy, zz, levels=20, cmap='coolwarm_r')
    ax.scatter(sample_excursionists, sample_tourists, color='limegreen')
    ax.set_title(f'{title} - Area = {area:.2f}')
    ax.set_xlim(left=0, right=x_max)
    ax.set_ylim(bottom=0, top=y_max)

fig, axs = plt.subplots(1, 3, figsize=(16, 5))
plot_scenario(axs[0], S_Base, 'Base')
plot_scenario(axs[1], S_Good_Weather, 'Good weather' )
plot_scenario(axs[2], S_Bad_Weather, 'Bad weather')
fig.colorbar(mappable=ScalarMappable(Normalize(0, 1), cmap='coolwarm_r'), ax=axs)
fig.show()
