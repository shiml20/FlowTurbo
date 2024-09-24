import torch as th


class ode:
    """ODE solver class"""
    def __init__(
        self,
        drift,
        *,
        t0,
        t1,
        num_steps
    ):
        assert t0 < t1, "ODE sampler has to be in forward time"

        self.drift = drift
        self.t = th.linspace(t0, t1, num_steps)
        self.t = self.t

    def sample(self, x, velocity_predictor, velocity_refiner, steps=['P', 'P'], **model_kwargs):
        
        device = x[0].device if isinstance(x, tuple) else x.device

        def _fn(t, x, model, refine_v=None):
            t = th.ones(x[0].size(0)).to(device) * t if isinstance(x, tuple) else th.ones(x.size(0)).to(device) * t
            model_output = self.drift(x, t, model, refine_v, **model_kwargs)
            return model_output
    
        t = self.t.to(device)
        delta_t = t[1] - t[0]
        
        xi = x
        samples = [xi]
        d_i_cache = None

        def one_step_heun(i, xi, model):
            d_i = _fn(t[i], xi, model)
            x_tilde_i_plus_1 = xi + delta_t * d_i
            d_i_plus_1 = _fn(t[i+1], x_tilde_i_plus_1, model)
            xi = xi + 1/2 * delta_t * (d_i + d_i_plus_1)
            return xi, d_i

        def one_step_pseudo_corrector(i, xi, model, d_i_cache=None):
            if d_i_cache is None:
                d_i = _fn(t[i], xi, model)
            else:
                d_i = d_i_cache
            x_tilde_i_plus_1 = xi + delta_t * d_i
            d_i_plus_1 = _fn(t[i+1], x_tilde_i_plus_1, model)
            xi = xi + 1/2 * delta_t * (d_i + d_i_plus_1)
            return xi, d_i, d_i_plus_1

        def one_step_refiner(i, xi, model, v):
            d_i = _fn(t[i], xi, model, v) + v
            x_tilde_i_plus_1 = xi + delta_t * d_i
            d_i_plus_1 = _fn(t[i+1], x_tilde_i_plus_1, model, d_i) + d_i
            xi = xi + 1/2 * delta_t * (d_i + d_i_plus_1)
            return xi, d_i


        for seq, i in enumerate(range(len(t)-1)):
            if steps[seq] == 'H':
                xi, d_i = one_step_heun(i, xi, velocity_predictor)
            elif steps[seq] == 'P':
                xi, d_i, d_i_cache = one_step_pseudo_corrector(i, xi, velocity_predictor, d_i_cache, begin=1)
                d_i = d_i_cache
            elif steps[seq] == 'R':
                xi, d_i = one_step_refiner(i, xi, velocity_refiner, d_i)

            samples.append(xi)
        
        return samples