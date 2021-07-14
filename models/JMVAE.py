from pixyz.distributions import Normal
from pixyz.losses import KullbackLeibler, Parameter
from pixyz.models import VAE
from torch import optim
import torch
from scipy.special import logsumexp


class JMVAE(object):
    def __init__(self, z_dim, optimizer_params, q_x, q_y, p_x, p_y, q=None, q_star_y=None, q_star_x=None, extra_modules=None, x_coef=1, y_coef=1, device=None):
        # prior model p(z)
        self.z_dim = z_dim
        self.prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                       var=["z"], features_shape=[z_dim], name="p_{prior}").to(device)
        if extra_modules is None:
            extra_modules = []
        self.p_x = p_x.to(device)
        self.p_y = p_y.to(device)
        self.p = self.p_x * self.p_y

        self.q = q.to(device)
        self.q_x = q_x.to(device)
        self.q_y = q_y.to(device)

        self.beta = Parameter("beta")

        likelihood_metrics = self.p_x.log_prob().expectation(self.q_y)

        self.name = 'JMVAE'

        alpha = 1

        kl = KullbackLeibler(self.q, self.prior)
        kl_x = KullbackLeibler(self.q, self.q_x)
        kl_y = KullbackLeibler(self.q, self.q_y)

        regularizer = kl + alpha*kl_x + alpha*kl_y

        self.model = VAE(self.q, self.p, other_distributions=extra_modules + [self.q_x, self.q_y],
                regularizer=self.beta*regularizer, optimizer=optim.Adam, optimizer_params=optimizer_params)

    def model_args(self, x, y, xu, yu, beta=1.0, is_supervised=1.0):
        return {"x1": x, "y1": y, "beta": beta}

    def eval_args(self, x, y):
        return {"y1": y, "x1": x, "beta": 1.0}

    def sample_z(self, y, sample_shape=None):
        if sample_shape:
            return self.q_y.sample({"y1": y}, sample_shape=[sample_shape], return_all=False)
        return self.q_y.sample({"y1": y}, return_all=False)

    def sample_z_from_x(self, x, sample_shape=None):
        if sample_shape:
            return self.q_x.sample({"x1": x}, sample_shape=[sample_shape], return_all=False)
        return self.q_x.sample({"x1": x}, return_all=False)

    def sample_z_all(self, x, y, sample_shape=None):
        if sample_shape:
            return self.q.sample({"x1": x, "y1": y}, sample_shape=[sample_shape], return_all=False)
        return self.q.sample({"x1": x, "y1": y}, return_all=False)

    def reconstruct_x(self, z):
        return self.p_x.sample_mean(z)

    def reconstruct_y(self, z):
        return self.p_y.sample_mean(z)

    def sample_prior(self, sample_shape=None):
        if sample_shape:
            return self.prior.sample(sample_shape=[sample_shape], return_all=False)
        return self.prior.sample(return_all=False)

    def get_number_of_parameters(self):
        result = 0
        for m in [self.q_x, self.q_y]:
            result += sum(p.numel() for p in m.parameters() if p.requires_grad)
        return result
