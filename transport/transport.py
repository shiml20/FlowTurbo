import torch as th
import numpy as np
import enum

from . import path
from .utils import EasyDict, log_state, mean_flat
from .integrators import ode

class ModelType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    VELOCITY = enum.auto()  # the model predicts v(x)

class PathType(enum.Enum):
    """
    Which type of path to use.
    """

    LINEAR = enum.auto()

class WeightType(enum.Enum):
    """
    Which type of weighting to use.
    """

    NONE = enum.auto()


class Transport:

    def __init__(
        self,
        *,
        model_type,
        path_type,
        loss_type,
    ):
        path_options = {
            PathType.LINEAR: path.ICPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()

    def prior_logp(self, z):
        '''
            Standard multivariate normal prior
            Assume z is batched
        '''
        shape = th.tensor(z.size())
        N = th.prod(shape[1:])
        _fn = lambda x: -N / 2. * np.log(2 * np.pi) - th.sum(x ** 2) / 2.
        return th.vmap(_fn)(z)


    def sample(self, x1):
        """Sampling x0 & t based on shape of x1 (if needed)
          Args:
            x1 - data point; [batch, *dim]
        """
        
        x0 = th.randn_like(x1)
        t0, t1 = 0, 1
        t = th.rand((x1.shape[0],)) * (t1 - t0) + t0
        t = t.to(x1)
        return t, x0, x1

    def training_losses(
        self, 
        model_student,
        model_teacher,
        x1, 
        model_kwargs=None
    ):
        """Loss for training the score model
        Args:
        - model: backbone model; could be score, noise, or velocity
        - x1: datapoint
        - model_kwargs: additional arguments for the model
        """
        if model_kwargs == None:
            model_kwargs = {}
        
        t, x0, x1 = self.sample(x1)
        t, xt, ut = self.path_sampler.plan(t, x0, x1)
        loss = []

        random_number = th.rand(1) * 0.12
        dt = random_number.item()
        dropout_prob = 0.1
        vt_teacher = model_teacher(xt, t, None, **model_kwargs)

        drop_ids = th.rand(model_kwargs['y'].shape[0], device=model_kwargs['y'].device) < dropout_prob
        model_kwargs['force_drop_ids'] = drop_ids
        x_t_plus_dt = xt + vt_teacher * dt
        vt_plus_dt_student = model_student(x_t_plus_dt, t + dt, v=vt_teacher, y=model_kwargs['y'], force_drop_ids=drop_ids) + vt_teacher
        vt_plus_dt_teacher = model_teacher(x_t_plus_dt, t + dt, None, **model_kwargs)
        _loss = mean_flat((vt_plus_dt_student - vt_plus_dt_teacher) ** 2)
        loss.append(_loss)

        terms = {}
        if self.model_type == ModelType.VELOCITY:
            terms['loss'] = loss[-1]
                
        return terms

    def get_drift(
        self
    ):
        """member function for obtaining the drift of the probability flow ODE"""
        
        def velocity_ode(x, t, model, v=None, **model_kwargs):
            model_output = model(x, t, v, **model_kwargs)
            return model_output

        drift_fn = velocity_ode
        
        def body_fn(x, t, model, v=None, **model_kwargs):
            model_output = drift_fn(x, t, model, v, **model_kwargs)
            return model_output

        return body_fn

   


class Sampler:
    """Sampler class for the transport model"""
    def __init__(
        self,
        transport,
    ):
        """Constructor for a general sampler; supporting different sampling methods
        Args:
        - transport: an tranport object specify model prediction & interpolant type
        """
        
        self.transport = transport
        self.drift = self.transport.get_drift()

    def sample_ode(
        self,
        *,
        num_steps=50,
    ):
        """returns a sampling function with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        - reverse: whether solving the ODE in reverse (data to noise); default to False
        """
        drift = self.drift

        t0, t1 = 0, 1

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
        )
        
        return _ode.sample
