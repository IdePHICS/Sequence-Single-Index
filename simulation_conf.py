import torch, math

SQRT2 = math.sqrt(2)

one_plus = lambda x: 1 + x
productory = lambda x: torch.prod(x, dim=-1)

# Function to be passed to the SequenceSingleIndex class
# ACHTUNG: they should act only on the last dimension of the input:
# z.shape = (..., B, L) 
# function(z).shape = (..., B)
ssi_function = {
    # linear
    "sum": lambda z:  torch.sum(z, dim=-1),
    "mean": lambda z: torch.mean(z, dim=-1),
    "sqrtsum": lambda z: math.sqrt(z.shape[-1]) * torch.sum(z, dim=-1),
    "inversesqrtsum": lambda z: 1/math.sqrt(z.shape[-1]) * torch.sum(z, dim=-1),
    "z1": lambda z: z[:,0],
    "sqrtz1": lambda z: math.sqrt(z.shape[-1]) * z[:,0],
    "2sum": lambda z: z[:,0]+z[:,1],
    # square
    "2product": lambda z: z[:,0]*z[:,1],
    "inversesqrtH2sum": lambda z: 1/math.sqrt(z.shape[-1]) * (z**2-1).sum(dim=-1)/SQRT2,
    "H2sum": lambda z: (z**2-1).sum(dim=-1)/SQRT2,
    "sqrtH2sum": lambda z: math.sqrt(z.shape[-1]) * (z**2-1).sum(dim=-1)/SQRT2,
    "inverseH2sum": lambda z: 1/z.shape[-1] * (z**2-1).sum(dim=-1)/SQRT2,
    "linearH2sum": lambda z: float(z.shape[-1])*(z**2-1).sum(dim=-1)/SQRT2,
    # extra
    "3product": lambda z: z[:,0]*z[:,1]*z[:,2],
    # null divergence
    "2difference": lambda z: z[:,0]-z[:,1],
    "2squaredifference": lambda z: (z[:,0]**2-z[:,1]**2)/math.sqrt(2),
    #staircase
    "basic_staircase": lambda z: z[:,0] + z[:,0]*z[:,1],
    "sqrtbasic_staircase": lambda z: math.sqrt(z.shape[-1])*(z[:,0] + z[:,0]*z[:,1]),
    "sqrtdouble_staircase": lambda z: math.sqrt(z.shape[-1])*(z[:,0] + z[:,0]*z[:,1] + z[:,0]*z[:,1]*z[:,2]),
    "complete_staircase": lambda z: z[:,0] * productory(one_plus(z[:,1:])),
}

activation = {
    "identity": lambda z: z,
    "relu": torch.nn.ReLU(),
    "tanh": torch.nn.Tanh(),
    "sigmoid": torch.nn.Sigmoid(),
    "square": lambda z: z**2,
    "H2": lambda z: (z**2-1)/SQRT2,
}

scaling_functions = {
    "none": lambda x: 1,
    "linear": lambda x: x,
    "square": lambda x: x**2,
    "sqrt": lambda x: math.sqrt(x),
    "cbrtsquare": lambda x: (x**2)**(1/3),
    "sqrttimelog": lambda x: math.sqrt(x) * math.log(x),
}

from sequence_single_index import StepCounterMetric, PopulationSquareLossMetric, OverlapMetric, StudentNormMetric, NormalizedOverlapComponentMetric, PositionalOverlapMetric

metrics = {
    "steps": StepCounterMetric,
    "population_square_loss": PopulationSquareLossMetric,
    "overlap": OverlapMetric,
    "student_norm": StudentNormMetric,
    "normalized_overlap_component": NormalizedOverlapComponentMetric,
    "positional_overlap": PositionalOverlapMetric,
}