from pixyz.distributions import Normal, Bernoulli, Categorical, ProductOfNormal

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

import numpy as np


max_len = 250

x_dim = 11000
z_dim = 300


class ReLUln100(nn.Module):
    def forward(self, x):
        return torch.clamp(x, min=0., max=np.log(101.))


class InferenceJoint(Normal):
    def __init__(self, coef=1):
        super(InferenceJoint, self).__init__(cond_var=["x1", "y1"], var=["z"], name="q_joint")

        self.encoder = nn.Sequential(
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

        self.conv1d1 = nn.Conv1d(max_len, 9, kernel_size=9)
        self.conv1d2 = nn.Conv1d(9, 9, kernel_size=9)
        self.conv1d3 = nn.Conv1d(9, 10, kernel_size=11)
        self.fc0 = nn.Linear(410, 435)

        self.fc_shared = nn.Linear(435 + int(1024 * coef), 1000)

        self.fc11 = nn.Linear(1000, z_dim)
        self.fc12 = nn.Linear(1000, z_dim)

    def forward(self, x1, y1):
        x = self.encoder(x1)

        h = F.relu(self.conv1d1(y1.float()))
        h = F.relu(self.conv1d2(h))
        h = F.relu(self.conv1d3(h))
        h = h.view(h.size(0), -1)
        h = F.selu(self.fc0(h))

        h = F.relu(self.fc_shared(torch.cat([x, h], 1)))

        return {"loc": self.fc11(h), "scale": F.softplus(self.fc12(h))}


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
    def __init__(self, coef=1):
        super(InferenceY, self).__init__(cond_var=["y1"], var=["z"], name="q2")

        self.conv1d1 = nn.Conv1d(max_len, 9, kernel_size=9)
        self.conv1d2 = nn.Conv1d(9, 9, kernel_size=9)
        self.conv1d3 = nn.Conv1d(9, 10, kernel_size=11)
        self.fc0 = nn.Linear(410, 435)
        self.fc11 = nn.Linear(435, z_dim)
        self.fc12 = nn.Linear(435, z_dim)

    def forward(self, y1):
        h = F.relu(self.conv1d1(y1.float()))
        h = F.relu(self.conv1d2(h))
        h = F.relu(self.conv1d3(h))
        h = h.view(h.size(0), -1)
        h = F.selu(self.fc0(h))
        return {"loc": self.fc11(h), "scale": F.softplus(self.fc12(h))}


# inference model q*1(z|y)
class InferenceY_missing(Normal):
    def __init__(self, coef=1):
        super(InferenceY_missing, self).__init__(cond_var=["y2"], var=["z"], name="q_star_1")

        self.conv1d1 = nn.Conv1d(max_len, 9, kernel_size=9)
        self.conv1d2 = nn.Conv1d(9, 9, kernel_size=9)
        self.conv1d3 = nn.Conv1d(9, 10, kernel_size=11)
        self.fc0 = nn.Linear(410, 435)
        self.fc11 = nn.Linear(435, z_dim)
        self.fc12 = nn.Linear(435, z_dim)

    def forward(self, y2):
        h = F.relu(self.conv1d1(y2.float()))
        h = F.relu(self.conv1d2(h))
        h = F.relu(self.conv1d3(h))
        h = h.view(h.size(0), -1)
        h = F.selu(self.fc0(h))
        return {"loc": self.fc11(h), "scale": F.softplus(self.fc12(h))}


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


# generative model p(y|z)
class GeneratorY(Categorical):
    def __init__(self):
        super(GeneratorY, self).__init__(cond_var=["z"], var=["y1"], name="p_y")
        h_size = 488

        self.fc2 = nn.Linear(z_dim, z_dim)
        self.gru = nn.GRU(z_dim, h_size, 3, batch_first=True)
        self.fc3 = nn.Linear(h_size, 67)

    def forward(self, z):
        z = F.selu(self.fc2(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, max_len, 1)
        out, h = self.gru(z)
        out_reshape = out.contiguous().view(-1, out.size(-1))
        y0 = F.softmax(self.fc3(out_reshape), dim=1)
        y = y0.contiguous().view(out.size(0), -1, y0.size(-1))
        return {"probs": y} 