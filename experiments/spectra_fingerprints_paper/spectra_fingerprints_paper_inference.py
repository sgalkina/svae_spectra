from pixyz.distributions import Normal, Bernoulli

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import numpy as np

x_dim = 11000
z_dim = 300
y_dim = 2149


class ReLUln100(nn.Module):
    def forward(self, x):
        return torch.clamp(x, min=0., max=np.log(101.))


class InferenceJoint(Normal):
    def __init__(self, n_inner=512, coef=1):

        super(InferenceJoint, self).__init__(cond_var=["x1", "y1"], var=["z"], name="q_joint")

        self.features_x = nn.Sequential(
            nn.Linear(in_features=x_dim, out_features=10000),
            nn.BatchNorm1d(10000),
            nn.ReLU(),
            nn.Linear(in_features=10000, out_features=5000),
            nn.BatchNorm1d(5000),
            nn.ReLU(),
            nn.Linear(in_features=5000, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.features_y = nn.Sequential(
            nn.Linear(in_features=y_dim, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=500),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 500, int(n_inner * coef)),
            nn.ReLU(),
            nn.Linear(int(n_inner * coef), z_dim * 2))
        self.n_latents = z_dim

    def forward(self, x1, y1):
        n_latents = self.n_latents
        x = self.features_x(x1)
        y = self.features_y(y1)
        x = self.classifier(torch.cat([x, y], 1))
        return {"loc": x[:, :n_latents], "scale": F.softplus(x[:, n_latents:])}


# inference model q1(z|x)
class InferenceX(Normal):
    def __init__(self):
        super(InferenceX, self).__init__(cond_var=["x1"], var=["z"], name="q1")

        self.encoder = nn.Sequential(
            nn.Linear(in_features=x_dim, out_features=10000),
            nn.BatchNorm1d(10000),
            nn.ReLU(),
            nn.Linear(in_features=10000, out_features=5000),
            nn.BatchNorm1d(5000),
            nn.ReLU(),
            nn.Linear(in_features=5000, out_features=1024*2),
            nn.BatchNorm1d(1024*2),
            nn.ReLU(),
        )
        self.fc11 = nn.Linear(1024, z_dim)
        self.fc12 = nn.Linear(1024, z_dim)

    def forward(self, x1):
        h = self.encoder(x1)
        log_var = F.softplus(self.fc12(h[:, 1024:]))
        return {"loc": self.fc11(h[:, :1024]), "scale": torch.exp(log_var / 2)}


# inference model q*2(z|x)
class InferenceX_missing(Normal):
    def __init__(self):
        super(InferenceX_missing, self).__init__(cond_var=["x2"], var=["z"], name="q_star_2")

        self.encoder = nn.Sequential(
            nn.Linear(in_features=x_dim, out_features=10000),
            nn.BatchNorm1d(10000),
            nn.ReLU(),
            nn.Linear(in_features=10000, out_features=5000),
            nn.BatchNorm1d(5000),
            nn.ReLU(),
            nn.Linear(in_features=5000, out_features=1024*2),
            nn.BatchNorm1d(1024*2),
            nn.ReLU(),
        )
        self.fc11 = nn.Linear(1024, z_dim)
        self.fc12 = nn.Linear(1024, z_dim)

    def forward(self, x2):
        h = self.encoder(x2)
        log_var = F.softplus(self.fc12(h[:, 1024:]))
        return {"loc": self.fc11(h[:, :1024]), "scale": torch.exp(log_var / 2)}


# inference model q2(z|y)
class InferenceY(Normal):
    def __init__(self, n_inner=512, coef=1):

        super(InferenceY, self).__init__(cond_var=["y1"], var=["z"], name="q2")

        self.N = 500
        self.encoder = nn.Sequential(
            nn.Linear(in_features=y_dim, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=int(self.N * 2 * coef)),
            nn.ReLU(),
        )
        self.fc11 = nn.Linear(int(self.N * coef), z_dim)
        self.fc12 = nn.Linear(int(self.N * coef), z_dim)

    def forward(self, y1):
        h = self.encoder(y1)
        return {"loc": self.fc11(h[:, :self.N]), "scale": F.softplus(self.fc12(h[:, self.N:]))}


# inference model q*1(z|y)
class InferenceY_missing(Normal):
    def __init__(self, n_inner=512, coef=1):

        super(InferenceY_missing, self).__init__(cond_var=["y2"], var=["z"], name="q_star_1")

        self.N = 500
        self.encoder = nn.Sequential(
            nn.Linear(in_features=y_dim, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=int(self.N * 2 * coef)),
            nn.ReLU(),
        )
        self.fc11 = nn.Linear(int(self.N * coef), z_dim)
        self.fc12 = nn.Linear(int(self.N * coef), z_dim)

    def forward(self, y2):
        h = self.encoder(y2)
        return {"loc": self.fc11(h[:, :self.N]), "scale": F.softplus(self.fc12(h[:, self.N:]))}


# generative model p(x|z)
class GeneratorX(Normal):
    def __init__(self):
        super(GeneratorX, self).__init__(cond_var=["z"], var=["x1"], name="p_x")
        self.decoder = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=3000),
            nn.ReLU(),
            nn.Linear(in_features=3000, out_features=10000),
            nn.ReLU(),
            nn.Linear(in_features=10000, out_features=15000),
            nn.ReLU(),
            nn.Linear(in_features=15000, out_features=x_dim),
            ReLUln100(),
        )

    def forward(self, z):
        h = self.decoder(z)
        return {"loc": h, "scale": 1.0}


# generative model p(x|z)
class GeneratorY(Bernoulli):
    def __init__(self):
        super(GeneratorY, self).__init__(cond_var=["z"], var=["y1"], name="p_y")
        self.decoder = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=y_dim),
        )

    def forward(self, z):
        h = self.decoder(z)
        return {"probs": torch.sigmoid(h)}


# inference model q1(z|x)
class SpectraToFingerprint(Bernoulli):
    def __init__(self):
        super(SpectraToFingerprint, self).__init__(cond_var=["x"], var=["y"])

        self.encoder = nn.Sequential(
            nn.Linear(in_features=x_dim, out_features=10000),
            nn.BatchNorm1d(10000),
            nn.ReLU(),
            nn.Linear(in_features=10000, out_features=5000),
            nn.BatchNorm1d(5000),
            nn.ReLU(),
            nn.Linear(in_features=5000, out_features=3000),
            nn.ReLU(),
            nn.Linear(in_features=3000, out_features=y_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        return {"probs": torch.sigmoid(h)}
