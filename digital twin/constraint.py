class Constraint:
    def __init__(self, capacity):
        self.capacity_index = capacity
        
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

class CapacityIndex:
    pass

class 