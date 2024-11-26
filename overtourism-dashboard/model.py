import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

class Context:    

    def __init__(
        self,
        name,
        data
    ):
        self.name = name
        self.data = data

class ModelViz:

    def __init__(
        self,
        name,
        x,
        y,
        constraints,
        contexts,
        values,
        diff
    ):
        self.name = name
        self.x = x
        self.y = y
        self.constraints = constraints
        self.contexts = contexts
        self.values = values
        self.diff = diff
        self.context_data = {} 
        self._build()


    def _build(self):
        xl = self.x['label']
        yl = self.y['label']

        for ctx in self.contexts:
            chart_data = compute_lines(self.x, self.y, self.constraints, ctx, self.values)
            chart_data.rename(columns={'x': xl, 'y': yl, 'label': 'Vincolo'}, inplace=True)
            points = compute_area(self.constraints, ctx, self.values)
            bound = self.values[self.x['id']] + self.values[self.y['id']]
            self.context_data[ctx.name] = ({"chart_data": chart_data, "points": points, "bound": bound, "context": ctx})

    def _diff_str(self):
        if not self.diff: 
            return ""
        res = list(map(lambda x: x + " " + (("+" + str(self.diff[x])) if self.diff[x] > 0 else str(self.diff[x])), self.diff.keys()))
        return ", ".join(res)
        
    def title(self):
        return f"{self.name}: {self._diff_str()}"
    
    def vis(self, ctx_name):
        xl = self.x['label']
        yl = self.y['label']

        ctx = self.context_data[ctx_name] 
        fig = px.line(ctx["chart_data"], x=xl, y=yl, color="Vincolo")
        fig.add_trace(go.Scatter(x=list(map(lambda p: p[0], ctx["points"])), 
                                 y=list(map(lambda p: p[1], ctx["points"])), 
                                 fill='tozeroy', showlegend=False, fillcolor='lightgrey', line_color='rgba(0,0,0,0)'))
        
        presenze = ctx["context"].data
        fig.add_trace(go.Scatter(name="presenze",
                                 x=presenze['value_x'],
                                 y=presenze['value_y'],
                                 mode='markers', showlegend=False, text=presenze['date']))
        
        fig.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor='grey', range=[0, ctx["bound"]])
        fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='grey', range=[0, ctx["bound"]])
        return fig

def compute_line(label, params, x_max, y_max, values):
    max = x_max + y_max
    r = list(map(lambda x: eval(x, {}, {"c": values}),params))
    try:
        x = 0
        p1 = [x, (r[2] - r[0]*x)/(r[1] if r[1] != 0 else 0.01)]
        if p1[1] > max: raise Exception            
    except:
        y = max
        p1 = [(r[2] - r[1]*y)/(r[0] if r[0] != 0 else 0.01), y]
    try:
        y = 0
        p2 = [(r[2] - r[1]*y)/(r[0] if r[0] != 0 else 0.01), y]
        if p2[0] > max: raise Exception
    except:
        x = max
        p2 = [x, (r[2] - r[0]*x)/(r[1] if r[1] != 0 else 0.01)]    
    return pd.DataFrame({ 'label': label, 'x': [p1[0], p2[0]], 'y': [p1[1], p2[1]]})

def compute_lines(x, y, constraints, context, values):
    frames = []
    for cstr in constraints:
        frames.append(compute_line(cstr['label'], cstr['r'], values[x['id']], values[y['id']], values))

    df = pd.concat(frames)
    return df
    
def compute_area(constraints, context, values):
    def intersect(r1, r2, matrix):
        div = (r1[0]*r2[1]-r2[0]*r1[1])
        if div == 0: div = 0.0000000001
        return [round((-r1[1]*r2[2]+r2[1]*r1[2])/div,2), round((-r1[2]*r2[0]+r2[2]*r1[0])/div,2)]
    def compute_matrix(constraints, values):
        res = []
        for cstr in constraints:
            r = list(map(lambda x: eval(x, {}, {"c": values}),cstr['r']))
            res.append(r)
        return res
        
    matrix = compute_matrix(constraints, values)
    min = None
    idx = 0
    for i in range(len(matrix)):
        p = intersect([1,0,0],matrix[i],matrix)
        if p[1] >= 0 and (min == None or min > p[1]):
            min = p[1]
            idx = i
    
    points = [[0,min]]
    
    processed = [idx]
    while len(processed) < len(matrix):
        min = None
        r1 = matrix[idx]
        point = None
        for i in range(len(matrix)):
            if i not in processed:
                r2 = matrix[i]
                p = intersect(r1, r2, matrix)
                if p[0] >= 0 and p[0] > points[-1][0] and (min == None or min > p[0]):
                    min = p[0]
                    idx = i
                    point = p
        if point is not None: points.append(point)
        processed.append(idx)    
    return points
