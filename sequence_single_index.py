import torch
import math

from typing import Union

# Decorator for handling batch dimension
def batch_decorator(func):
    def wrapper(self, x, *args, **kwargs):
        batch_dim = True
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            batch_dim = False
        y = func(self, x, *args, **kwargs)
        if not batch_dim:
            y = y.squeeze(0)
        return y
    return wrapper

class SequenceSingleIndex(torch.nn.Module):
    def __init__(
            self,
            hidden_direction: torch.tensor,
            # Function to be applied on a low-dimensional tensor of shape (..., L)
            function: torch.nn.Module,
            require_grad: bool = False
        ):
        super().__init__()
        assert len(hidden_direction.shape) == 1
        self.hidden_direction = torch.nn.Parameter(hidden_direction, requires_grad=require_grad)
        self.function = function
    
    @batch_decorator
    def forward(self, x: torch.tensor):
        # x has shape (B, L, D) with
        # B: batch size
        # L: sequence length
        # D: hidden_direction dimension
        B, L, d = x.shape
        x = x.view(B*L, d)
        x = torch.matmul(x, self.hidden_direction)
        x = x.view(B, L)
        x = self.function(x)
        return x
    
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d: int, max_L: int):
        super().__init__()
        self.d = d
        self.max_L = max_L

    def __call__(self, x: torch.tensor):
        # x has shape (B, L, D) with
        # B: batch size
        # L: sequence length
        # d: hidden_direction dimension
        B, L, d = x.shape
        assert d == self.d or self.d == -1
        assert L <= self.max_L or self.max_L == -1
        return x + self.positional_tensor(L).unsqueeze(0) / math.sqrt(self.d)  # shape (B, L, d) + shape (1, L, d)
    
    def positional_tensor(self, L: int):
        raise NotImplementedError("PositionalEncoding is a base class! is not implemented")

    
class NoPositionalEncoding(PositionalEncoding):
    def __init__(self):
        super().__init__(-1, -1)
    
    def __call__(self, x: torch.tensor):
        return x
    
class OppositePairedPositionalEncoding(PositionalEncoding):
    def __init__(self, d: int, norm: float = 1.0, orthogonal_to: torch.tensor = None, seed: Union[None, int] = None):
        super().__init__(d, 2)
    
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        positional_direction = torch.randn(d, generator=generator)

        if orthogonal_to is not None:
            assert orthogonal_to.shape[-1] == d
            if len(orthogonal_to.shape) == 1:
                orthogonal_to = orthogonal_to.unsqueeze(0)
            orthogonal_to /= torch.norm(orthogonal_to, dim=-1, keepdim=True)
            positional_direction = positional_direction - torch.einsum('ij,j,ik->k', orthogonal_to, positional_direction, orthogonal_to)
        positional_direction /= torch.norm(positional_direction) / norm
        self._positional_tensor = torch.nn.Parameter(
            torch.stack((
                positional_direction,
                -positional_direction
            ), dim=0),
            requires_grad=False
        )

    def positional_tensor(self, L: int):
        return self._positional_tensor[:L, :]

class AttentionReduction():
    def __init__(self):
        pass

    def __call__(self, x: torch.tensor):
        raise NotImplementedError("AttentionReduction is a base class! __call__ is not implemented")
    
class NoAttentionReduction(AttentionReduction):
    def __call__(self, x):
        return x
    
class ProjectionAttentionReduction(AttentionReduction):
    def __init__(self, a_left: torch.tensor, a_right: torch.tensor):
        super().__init__()
        self.a_left = torch.nn.Parameter(a_left, requires_grad=False)
        self.a_right = torch.nn.Parameter(a_right, requires_grad=False)

    def __call__(self, a: torch.tensor):
        # a has shape (B, L, L) with
        # B: batch size
        # L: sequence length
        a = torch.einsum('i, bij, j->b', self.a_left, a, self.a_right) # shape (B,)
        return a

class TraceAttentionReduction(AttentionReduction):
    def __call__(self, a: torch.tensor):
        # a has shape (B, L, L) with
        # B: batch size
        # L: sequence length
        L = a.shape[-1]
        a = 1/math.sqrt(L) * torch.einsum('bii->b', a) # shape (B,)
        return a
    
class SingleIndexAttention(torch.nn.Module):
    def __init__(
            self,
            hidden_direction: torch.tensor, # shape (D,)
            require_grad: bool = False,
            positional_encoding: PositionalEncoding = NoPositionalEncoding(),
            reduction: AttentionReduction = NoAttentionReduction(),
        ):
        super().__init__()
        self.hidden_direction = torch.nn.Parameter(hidden_direction, requires_grad=require_grad)
        self.positional_encoding = positional_encoding
        self.reduction = reduction
        
   
    @batch_decorator
    def forward(self, x: torch.tensor):
        # x has shape (B, L, D) with
        # B: batch size
        # L: sequence length
        # D: hidden_direction dimension
        B, L, d = x.shape
        x = self.positional_encoding(x) # Add the positional encoding
        x = x.view(B*L, d)
        x = torch.matmul(x, self.hidden_direction)
        x = x.view(B, L)
        a = torch.einsum('bi,bj->bij', x, x) # shape (B, L, L)
        a = torch.softmax(a, dim=-1) # shape (B, L, L)
        a = self.reduction(a)
        return a
    
class PositionalSemanticTransitionModel(SingleIndexAttention):
    def __init__(
            self,
            hidden_direction: torch.tensor, # shape (D,)
            omega: float,
            A: torch.tensor, # shape (L, L)
        ):
        super().__init__(
            hidden_direction,
            require_grad=False,
            positional_encoding=NoPositionalEncoding(),
            reduction=NoAttentionReduction(),
        )
        assert 0. <= omega <= 1.0
        self.omega = omega
        self.A = A.detach().clone() # shape (L, L)
        self.A = torch.nn.Parameter(self.A.unsqueeze(0), requires_grad=False) # shape (1, L, L)

    @batch_decorator
    def forward(self, x: torch.tensor):
        return (
            (1-self.omega) * super().forward(x) +
            self.omega * self.A
        ) # shape (B, L, L)
        
    
class PlainFeedForward(torch.nn.Module):
    def __init__(
            self,
            hidden_direction: torch.tensor, # shape (D*width)
            width: int,
            activation: torch.nn.Module,
            require_grad: bool = False
        ):
        super().__init__()
        self.hidden_direction = torch.nn.Parameter(hidden_direction, requires_grad=require_grad)
        if width > 1:
            raise NotImplementedError("Only width = 1 is implemented")
        self.activation = activation

    @batch_decorator
    def forward(self, x: torch.tensor):
        # x has shape (B, L, D) with
        # B: batch size
        # L: sequence length
        # D: hidden_direction dimension

        B, L, d = x.shape
        x = x.view(B, L*d)
        x = 1/math.sqrt(L) * torch.matmul(x, self.hidden_direction)
        x = x.view(B,)
        return self.activation(x)
    
class TiedFeedForward(torch.nn.Module):
    def __init__(
            self,
            hidden_direction: torch.tensor, # shape (D,)
            width: int,
            activation: torch.nn.Module,
            require_grad: bool = False
        ):
        super().__init__()
        self.hidden_direction = torch.nn.Parameter(hidden_direction, requires_grad=require_grad)
        if width > 1:
            raise NotImplementedError("Only width = 1 is implemented")
        self.activation = activation

    @batch_decorator
    def forward(self, x: torch.tensor):
        # x has shape (B, L, D) with
        # B: batch size
        # L: sequence length
        # D: hidden_direction dimension
        B, L, d = x.shape
        x = x.view(B*L, d)
        x = torch.matmul(x, self.hidden_direction)
        x = x.view(B, L) # shape (B, L)
        x = 1/math.sqrt(L) * torch.sum(x, dim=1) # shape (B,)
        return self.activation(x)
    

# Optimizer class that implements plain SGD, but keeps the norm of the weights constant
class NormalizedSGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float =0.01, eps: float = 1e-12):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, eps=eps)
        super(NormalizedSGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']

            for param in group['params']:
                if param.grad is None:
                    continue
                
                previous_norm = torch.norm(param.data, dim=-1, keepdim=True)
                param.add_(param.grad, alpha=-lr)
                actual_norm = torch.norm(param.data, dim=-1, keepdim=True)
                param.data /= actual_norm + eps
                param.data *= previous_norm

class FrobeniusNormLoss(torch.nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        if reduction not in ['mean', 'sum']:
            raise ValueError(f"Invalid reduction: {reduction}. Must be 'mean' or 'sum'.")
    
    def forward(self, target: torch.tensor, output: torch.tensor):
        assert target.shape == output.shape, f"Target shape {target.shape} does not match output shape {output.shape}"
        diff = target - output
        loss = torch.norm(diff, p='fro', dim=(-2,-1))
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)

class BaseInitialCondition():
    def _random_direction(self):
        rd = torch.randn(self.d, generator=self._generator)
        rd /= torch.norm(rd) / math.sqrt(self.d)
        return rd
    
    def __init__(self, d: int, seed: Union[None, int] = None):
        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)
        self.d = d
        self._hidden_direction = self._random_direction()

    def teacher(self,):
        # return a copy of the hidden direction
        return self._hidden_direction.detach().clone()
    
class RandomInitialCondition(BaseInitialCondition):
    def __init__(self, d: int, seed: Union[None, int] = None, normalized: bool = True):
        super().__init__(d, seed)
        self.normalized = normalized
    def student(self, L: int = 1):
        # Generate a random direction
        random_student = torch.stack([self._random_direction() for _ in range(L)])
        # Normalize the random direction if required
        if self.normalized:
            student = random_student / torch.norm(random_student, dim=-1, keepdim=True) * math.sqrt(self.d)
        else:
            student = random_student
        return student.view(L * self.d)

class OverlappedInitialCondition(BaseInitialCondition):
    
    def __init__(self, d: int, overlap: Union[None, int] = None, seed: Union[None, int] = None, random_sign: bool = False):
        super().__init__(d, seed)
        if overlap is None:
            overlap = 1/math.sqrt(d)
        self.overlap = overlap
        self.random_sign = random_sign
    
    def student(self, L: int = 1):
        random_student = torch.stack([self._random_direction() for _ in range(L)])
        random_overlap = torch.matmul(random_student, self._hidden_direction) / self.d
        orthogonal_student = random_student - random_overlap.view(L,1) * self._hidden_direction.unsqueeze(0).repeat(L, 1)
        orthonormal_student = orthogonal_student / torch.norm(orthogonal_student, dim=-1, keepdim=True) * math.sqrt(self.d)

        if self.random_sign:
            overlap_sign = torch.sign(random_overlap)
        else:
            overlap_sign = torch.ones_like(random_overlap)

        unnormalized_student = (1 - self.overlap) * orthonormal_student + self.overlap * torch.diag(overlap_sign) @ torch.tile(self._hidden_direction.unsqueeze(0), (L, 1))

        student = unnormalized_student / torch.norm(unnormalized_student, dim=-1, keepdim=True) * math.sqrt(self.d)
        return student.view(L * self.d)
    
class BaseMetric():
    def __init__(self, teacher: torch.nn.Module, student: torch.nn.Module, name: str = None):
        self.teacher = teacher
        self.student = student
        self._memory = list()
        self._custom_name = str(name)

    def _compute(self, **kwargs):
        raise NotImplementedError("BaseMetric is an abstract class")

    @torch.no_grad()
    def __call__(self, **kwargs):
        self._memory.append(self._compute(**kwargs))
    
    @property
    def data(self):
        return torch.tensor(self._memory)
    
    @property
    def name(self):
        return self._custom_name
    
class OverlapMetric(BaseMetric):
    def __init__(self, teacher: torch.nn.Module, student: torch.nn.Module, norm: str = 'l2', signed: bool = False, **kwargs):
        super().__init__(teacher, student, name='m_metric')
        self.d = self.teacher.hidden_direction.shape[0]
        self.signed = signed
        if isinstance(self.student, PlainFeedForward):
            self._sequence_length_factor = self.student.hidden_direction.shape[0] // self.d
        else:
            self._sequence_length_factor = 1

        match norm:
            case 'l2':
                self._p = 2
                self._norm_operator = lambda x: torch.norm(x, p=2)
            case 'l1':
                self._p = 1
                self._norm_operator = lambda x: torch.norm(x, p=1)
            case None:
                self._p = None
                self._norm_operator = lambda x: x

    @torch.no_grad()
    def _compute(self, **kwargs):
        if isinstance(self.student, PlainFeedForward):
            student_hidden_direction = self.student.hidden_direction.view(self._sequence_length_factor, self.d)
        else:
            student_hidden_direction = self.student.hidden_direction.view(-1, self.d)
        # Compute the overlap between the teacher and student
        overlap = torch.matmul(student_hidden_direction, self.teacher.hidden_direction) / self.d # shape (sequence_length_factor, )
        try:
            assert overlap.shape[0] == self._sequence_length_factor
        except AssertionError:
            raise AssertionError(f"Overlap shape {overlap.shape} does not match sequence length factor {self._sequence_length_factor}")
        
        if self._p is None:
            return overlap
        
        m = self._norm_operator(overlap).squeeze() / math.pow(self._sequence_length_factor, 1/self._p)

        # If the overlap was a scalar, we can preserve the sign
        if self.signed and overlap.shape[0] == 1:
            m *= torch.sign(overlap).squeeze()

        return m
    
class NormalizedOverlapComponentMetric(OverlapMetric):
    def __init__(self, teacher: torch.nn.Module, student: torch.nn.Module, component: int = 0, **kwargs):
        super().__init__(teacher, student, norm=None, **kwargs)
        self._custom_name = f'normalized_overlap_component_{component}'
        self._component = component
    
    @torch.no_grad()
    def _compute(self, **kwargs):
        overlap = super()._compute(**kwargs)
        return overlap[self._component] / torch.norm(overlap, p=2)

    
class PopulationSquareLossMetric(BaseMetric):
    def __init__(self, teacher: torch.nn.Module, student: torch.nn.Module, L: int, test_size: Union[None, int] = None, **kwargs):
        super().__init__(teacher, student, name='loss_metric')
        self.d = self.teacher.hidden_direction.shape[0]
        self.L = L
        self.test_size = test_size
        if test_size is None:
            self.test_size = 20*self.d
        
        self._testdata = torch.randn(self.test_size, self.L, self.d) / math.sqrt(self.d)

    @torch.no_grad()
    def _compute(self, **kwargs):
        single_sample_loss = 1/2 * (self.teacher(self._testdata) - self.student(self._testdata))**2
        try:
            assert single_sample_loss.shape == (self.test_size,)
        except AssertionError:
            raise AssertionError(f"Single sample loss shape {single_sample_loss.shape} does not match test size {self.test_size}")
        
        return torch.mean(single_sample_loss).item()
        
class StepCounterMetric(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(None, None, name='steps')

    @torch.no_grad()
    def _compute(self, step, **kwargs):
        return step
    
class StudentNormMetric(BaseMetric):
    """
    Norm of the student weight, divided by sqrt(d).
    
    In the case of PlainFeedForward, the norm is also divided by sqrt(L) for being comparable to the tied network.
    """
    def __init__(self, teacher: torch.nn.Module, student: torch.nn.Module, **kwargs):
        super().__init__(teacher, student, name='student_norm')
        self.d = self.teacher.hidden_direction.shape[0]
        if isinstance(self.student, PlainFeedForward):
            self._sequence_length_factor = self.student.hidden_direction.shape[0] // self.d
        else:
            self._sequence_length_factor = 1

    @torch.no_grad()
    def _compute(self, **kwargs):
        student_hidden_direction = self.student.hidden_direction.view(self._sequence_length_factor, self.d)

        return torch.norm(
            student_hidden_direction,
            p=2
        ) / math.sqrt(self.d*self._sequence_length_factor)
    
class PositionalOverlapMetric(BaseMetric):
    def __init__(self, teacher: torch.nn.Module, student: torch.nn.Module, **kwargs):
        super().__init__(teacher, student, name='positional_overlap')
        self.d = self.teacher.hidden_direction.shape[0]
        assert isinstance(self.student, SingleIndexAttention), "PositionalOverlapMetric only works with SingleIndexAttention"
        assert isinstance(self.student.positional_encoding, OppositePairedPositionalEncoding), "PositionalOverlapMetric only works with OppositePairedPositionalEncoding"

    @torch.no_grad()
    def _compute(self, **kwargs):
        return torch.dot(
            self.student.hidden_direction,
            self.student.positional_encoding.positional_tensor(1).squeeze(0)
        )/ math.sqrt(self.d)

from torch.utils.data import IterableDataset

class GaussianTeacherDataset(IterableDataset):
    """
    Custom PyTorch IterableDataset that generates random data and processes it through a teacher model.
    
    Args:
        teacher (torch.nn.Module): The teacher model to generate labels
        L (int): Length of the sequence (L)
        feature_dim (int): Feature dimension (d)
        device (torch.device): Device to use for computation
        seed (int, optional): Random seed for reproducibility
    """
    def __init__(self, teacher: torch.nn.Module, L: int, seed=None, number_of_samples: int = 1000):
        super().__init__()
        self.teacher = teacher
        self.L = L
        self.d = teacher.hidden_direction.shape[0]
        self.device = teacher.hidden_direction.device

        self.number_of_samples = number_of_samples if number_of_samples > 0 else None
        
        # Set teacher to evaluation mode
        self.teacher.eval()
        
        # Initialize generator with seed if provided
        self._generator = torch.Generator(device=self.device)
        if seed is not None:
            self._generator.manual_seed(seed)
    
    def __iter__(self):
        self._generated_samples = 0
        return self
    
    def __next__(self):
        if self.number_of_samples is not None and self._generated_samples >= self.number_of_samples:
            raise StopIteration
        self._generated_samples += 1

        x = torch.randn(
            self.L, 
            self.d, 
            device=self.device, 
            generator=self._generator
        ) / math.sqrt(self.d)
        
        with torch.no_grad():
            y = self.teacher(x)
        
        return x, y

if __name__ == "__main__":
    d = 1000
    L = 2

    # Set default device to cuda, mps, or cpu in this order of preference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu" if torch.backends.mps.is_available() else "cpu")

    initial_condition = OverlappedInitialCondition(d=d, overlap=0, seed = 8205)

    A = torch.softmax(torch.tensor([[1.,-1.],[-1.,1.]]), dim=-1)
    teacher = PositionalSemanticTransitionModel(
        hidden_direction=initial_condition.teacher(),
        omega=.65,
        A=A,
    ).to(device)
    student_ic = initial_condition.student()
    pe = OppositePairedPositionalEncoding(
        d=d, norm=1.0,
        orthogonal_to=torch.stack((student_ic, initial_condition.teacher()), dim=0),
        seed = 20573
    )
    student = SingleIndexAttention(
        # hidden_direction=math.sqrt(0.8)*student_ic+math.sqrt(0.2)*pe.positional_tensor(1).squeeze(0)*math.sqrt(d),
        hidden_direction=student_ic,
        require_grad=True,
        positional_encoding=pe,
    ).to(device)


    loss = FrobeniusNormLoss().to(device)
    optimizer = NormalizedSGD(student.parameters(), lr=5e-2)

    loss_metric = list()
    m = OverlapMetric(teacher, student, signed=True)
    e = PositionalOverlapMetric(teacher, student)

    m()
    e()

    from tqdm import tqdm

    n_steps = int(3e6)
    dataset = GaussianTeacherDataset(teacher, L, seed=34795, number_of_samples=n_steps)
    for xy in tqdm(dataset, total=n_steps):
        x, y = xy
        y_hat = student(x)
        l = loss(y, y_hat)

        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        
        loss_metric.append(l.item())
        m()
        e()

    m_metric = m.data
    e_metric = e.data


    import matplotlib.pyplot as plt
    plt.plot(m_metric, label='Overlap')
    plt.axhline(1/torch.sqrt(torch.tensor(d)).item(), color='r')
    plt.plot(e_metric, label='Positional overlap')
    plt.xlabel('Steps')
    plt.ylabel('Metric Value')
    plt.title('Overlap and Positional Overlap Metrics')
    plt.legend()
    plt.show()
