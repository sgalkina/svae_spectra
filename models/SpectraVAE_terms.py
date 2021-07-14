from pixyz.distributions import Normal, ProductOfNormal
from pixyz.losses import KullbackLeibler, Parameter
from pixyz.models import Model
from torch import optim
import torch
from scipy.special import logsumexp
from models.utils import unsupervised_distr


# if torch.cuda.is_available():
#     device = "cuda:2"
# else:
#     device = "cpu"
#
# # device = "cpu"


class SpectraVAE(object):
    def __init__(self, z_dim, optimizer_params, q_x, q_y, p_x, p_y, q=None, q_star_y=None, q_star_x=None, extra_modules=None, x_coef=1, y_coef=1, device=None):
        self.sample_shape = 1000
        self.z_dim = z_dim
        if extra_modules is None:
            extra_modules = []
        self.prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                       var=["z"], features_shape=[z_dim], name="p_{prior}").to(device)

        self.p_x = p_x.to(device)
        self.p_y = p_y.to(device)
        self.p = self.p_x * self.p_y

        self.beta = Parameter("beta")
        self.is_supervised = Parameter("is_supervised")

        self.q1_x = q_x.to(device)
        self.q2_y = q_y.to(device)
        self.q_star_1_y = q_star_y.to(device)
        self.q_star_2_x = q_star_x.to(device)

        self.q = ProductOfNormal([self.q1_x, self.q2_y], name="q").to(device)
        self.q1 = ProductOfNormal([self.q1_x, self.q_star_2_x], name="q_x_only").to(device)
        self.q2 = ProductOfNormal([self.q2_y, self.q_star_1_y], name="q_y_only").to(device)

        self.name = 'SpectraVAE_terms'

        self.p_x_u, self.x_vars = unsupervised_distr(self.p_x)
        self.p_y_u, self.y_vars = unsupervised_distr(self.p_y)

        self.q1_x_u, self.x_q_vars = unsupervised_distr(self.q1_x)
        self.q_star_2_x_u, self.x_q_star_vars = unsupervised_distr(self.q_star_2_x)

        self.q2_y_u, self.y_q_vars = unsupervised_distr(self.q2_y)
        self.q_star_1_y_u, self.y_q_star_vars = unsupervised_distr(self.q_star_1_y)

        self.p_x_u = self.p_x_u.to(device)
        self.p_y_u = self.p_y_u.to(device)
        self.q1_x_u = self.q1_x_u.to(device)
        self.q_star_2_x_u = self.q_star_2_x_u.to(device)
        self.q2_y_u = self.q2_y_u.to(device)
        self.q_star_1_y_u = self.q_star_1_y_u.to(device)

        self.q1_u = ProductOfNormal([self.q1_x_u, self.q_star_2_x_u], name="q_x_u_only").to(device)
        self.q2_u = ProductOfNormal([self.q2_y_u, self.q_star_1_y_u], name="q_y_u_only").to(device)

        loss_supervised = -x_coef*self.p_x.log_prob().expectation(self.q) - y_coef*self.p_y.log_prob().expectation(self.q) + \
                          -x_coef*self.p_x.log_prob().expectation(self.q1) - y_coef*self.p_y.log_prob().expectation(self.q2) + \
                          self.beta * KullbackLeibler(self.q, self.q1) + self.beta * KullbackLeibler(self.q, self.q2) + \
                          self.beta * KullbackLeibler(self.q1, self.prior) + self.beta * KullbackLeibler(self.q2, self.prior)
        loss_unsupervised_x = -x_coef*self.p_x_u.log_prob().expectation(self.q1_u) + self.beta*KullbackLeibler(self.q1_u, self.prior)
        loss_unsupervised_y = -y_coef*self.p_y_u.log_prob().expectation(self.q2_u) + self.beta*KullbackLeibler(self.q2_u, self.prior)

        self.model = Model(loss=self.is_supervised*(loss_supervised.mean() + loss_unsupervised_y.mean()) + loss_unsupervised_x.mean(),
                      distributions=extra_modules + [self.p_x, self.p_y, self.q1_x, self.q2_y, self.q_star_1_y, self.q_star_2_x,
                                     self.q1, self.q2, self.q],
                      optimizer=optim.Adam,
                      optimizer_params=optimizer_params)

    def loss_components(self, x, y, x_u, y_u):
        results = []
        for loss in [
            -self.p_x.log_prob().expectation(self.q).mean().eval({'x1': x, 'y1': y, 'x2': x, 'y2': y}),
            -self.p_y.log_prob().expectation(self.q).mean().eval({'x1': x, 'y1': y, 'x2': x, 'y2': y}),
            KullbackLeibler(self.q, self.q1).mean().eval({'x1': x, 'y1': y, 'x2': x, 'y2': y}),
            KullbackLeibler(self.q, self.q2).mean().eval({'x1': x, 'y1': y, 'x2': x, 'y2': y}),
            KullbackLeibler(self.q1, self.prior).mean().eval({'x1': x, 'x2': x}),
            KullbackLeibler(self.q2, self.prior).mean().eval({'y1': y, 'y2': y}),
            -self.p_x_u.log_prob().expectation(self.q1_u).mean().eval({'x1_u': x_u, 'x2_u': x_u}),
            KullbackLeibler(self.q1_u, self.prior).mean().eval({'x1_u': x_u, 'x2_u': x_u}),
            -self.p_y_u.log_prob().expectation(self.q2_u).mean().eval({'y1_u': y_u, 'y2_u': y_u}),
            KullbackLeibler(self.q2_u, self.prior).mean().eval({'y1_u': y_u, 'y2_u': y_u}),
        ]:
            results.append(float(loss))
        return results

    def log_likelihood_function(self, x, sample_size=1000):
        result = []
        for i in range(x.shape[0]):
            z = self.q1.sample({'x1': x[i].unsqueeze(0), 'x2': x[i].unsqueeze(0)}, sample_shape=[sample_size])['z']
            p_probs = self.prior.log_prob().eval({'z': z})

            a = self.q1.get_params({'x1': x[i].unsqueeze(0), 'x2': x[i].unsqueeze(0)})
            expert = Normal(loc=a['loc'], scale=a['scale'],
                            var=["z"], features_shape=[1, self.z_dim], name="expert").to(device)
            q_probs = expert.log_prob().eval({'z': z, 'x1': x[i].unsqueeze(0), 'x2': x[i].unsqueeze(0)})

            p_x_z_probs = self.p_x.log_prob().eval({'z': z, 'x1': x[i].unsqueeze(0), 'x2': x[i].unsqueeze(0)})
            a = logsumexp((p_x_z_probs + p_probs - q_probs).detach().cpu().numpy(), axis=-1)
            result.append(float(a))
        return result

    def log_likelihood_function_x_y(self, x, y, sample_size=1000):
        result = []
        for i in range(x.shape[0]):
            z = self.q.sample({'x1': x[i].unsqueeze(0), 'y1': y[i].unsqueeze(0)}, sample_shape=[sample_size])['z']
            p_probs = self.prior.log_prob().eval({'z': z})
            p_x_z_probs = self.p_x.log_prob().eval({'z': z, 'x1': x[i].unsqueeze(0)})

            a = self.q.get_params({'x1': x[i].unsqueeze(0), 'y1': y[i].unsqueeze(0)})
            expert = Normal(loc=a['loc'], scale=a['scale'],
                            var=["z"], features_shape=[1, self.z_dim], name="expert").to(device)
            q_probs = expert.log_prob().eval({'z': z, 'x1': x[i].unsqueeze(0), 'y1': y[i].unsqueeze(0)})

            a = logsumexp((p_x_z_probs + p_probs - q_probs).detach().cpu().numpy(), axis=-1)
            result.append(float(a))
        return result

    def test_joint_log_likelihood(self, x, y, sample_size=1000):
        result = []
        for i in range(x.shape[0]):
            z = self.q.sample({'x1': x[i].unsqueeze(0), 'y1': y[i].unsqueeze(0)}, sample_shape=[sample_size])['z']
            p_probs = self.prior.log_prob().eval({'z': z})
            p_x_z_probs = self.p_x.log_prob().eval({'z': z, 'x1': x[i].unsqueeze(0)})
            p_y_z_probs = self.p_y.log_prob().eval({'z': z, 'y1': y[i].unsqueeze(0)})

            a = self.q.get_params({'x1': x[i].unsqueeze(0), 'y1': y[i].unsqueeze(0)})
            expert = Normal(loc=a['loc'], scale=a['scale'],
                            var=["z"], features_shape=[1, self.z_dim], name="expert").to(device)
            q_probs = expert.log_prob().eval({'z': z, 'x1': x[i].unsqueeze(0), 'y1': y[i].unsqueeze(0)})

            a = logsumexp((p_x_z_probs + p_y_z_probs + p_probs - q_probs).detach().cpu().numpy(), axis=-1)
            result.append(float(a))
        return result

    def model_args(self, x, y, xu, yu, beta=1.0, is_supervised=1.0):
        return {"x1": x, "y1": y, "x2": x, "y2": y, "beta": beta, "x1_u": xu, "x2_u": xu, "y1_u": yu, "y2_u": yu,
                'is_supervised': is_supervised}

    def eval_args(self, x, y):
        return {"x1": x, "y1": y, "x2": x, "y2": y, "beta": 1, "x1_u": x, "x2_u": x, "y1_u": y, "y2_u": y,
                'is_supervised': 1}

    def loss_components_colors(self, y_rgb, x_src, x_tgt, x_len, y_rgb_u, x_src_u, x_tgt_u, x_len_u):
        results = []
        eval_args_both = {
            "x1": x_src, 'x1_ls': x_len,
            'x1_p': x_src, 'x1_p_ls': x_len,
            "y1": y_rgb,

            "x2": x_src, 'x2_ls': x_len,
            'x2_p': x_src, 'x2_p_ls': x_len,
            "y2": y_rgb,

            'xt': x_tgt,
        }
        eval_args_x_u = {
            'x1_u': x_src_u, 'x1_ls_u': x_len_u,
            'x1_p_u': x_src_u, 'x1_p_u_ls': x_len_u,
            'x2_u': x_src_u, 'x2_ls_u': x_len_u,
            'x2_p_u': x_src_u, 'x2_p_u_ls': x_len_u,
            'xt_u': x_tgt_u,
        }
        eval_args_y_u = {
            'y1_u': y_rgb_u,
            'y2_u': y_rgb_u
        }
        for loss in [
            -self.p_x.log_prob().expectation(self.q).mean().eval(eval_args_both),
            -self.p_y.log_prob().expectation(self.q).mean().eval(eval_args_both),
            KullbackLeibler(self.q, self.q1).mean().eval(eval_args_both),
            KullbackLeibler(self.q, self.q2).mean().eval(eval_args_both),
            -self.p_x_u.log_prob().expectation(self.q1_u).mean().eval(eval_args_x_u),
            KullbackLeibler(self.q1_u, self.prior).mean().eval(eval_args_x_u),
            -self.p_y_u.log_prob().expectation(self.q2_u).mean().eval(eval_args_y_u),
            KullbackLeibler(self.q2_u, self.prior).mean().eval(eval_args_y_u),
        ]:
            results.append(float(loss))
        return results

    def log_likelihood_function_colors(self, y_rgb, x_src, x_tgt, x_len, sample_size=1000):
        result = []
        for i in range(x_src.shape[0]):
            eval_args_x = {
                "x1": x_src[i].unsqueeze(0), 'x1_ls': x_len[i].unsqueeze(0),
                'x1_p': x_src[i].unsqueeze(0), 'x1_p_ls': x_len[i].unsqueeze(0),
                "x2": x_src[i].unsqueeze(0), 'x2_ls': x_len[i].unsqueeze(0),
                'x2_p': x_src[i].unsqueeze(0), 'x2_p_ls': x_len[i].unsqueeze(0),
                'xt': x_tgt,
            }
            z = self.q1.sample(eval_args_x, sample_shape=[sample_size])['z']
            eval_args_xz = {'z': z}
            eval_args_xz.update(eval_args_x)
            p_probs = self.prior.log_prob().eval({'z': z})
            p_x_z_probs = self.p_x.log_prob().eval(eval_args_xz)
            q_probs = self.q1.log_prob().eval(eval_args_xz)
            a = logsumexp((p_x_z_probs + p_probs - q_probs).detach().cpu().numpy(), axis=-1)
            result.append(float(a))
        return result

    def test_joint_log_likelihood_colors(self, y_rgb, x_src, x_tgt, x_len, sample_size=1000):
        result = []
        for i in range(x_src.shape[0]):
            eval_args_xy = {
                "x1": x_src[i].unsqueeze(0), 'x1_ls': x_len[i].unsqueeze(0),
                'x1_p': x_src[i].unsqueeze(0), 'x1_p_ls': x_len[i].unsqueeze(0),
                "y1": y_rgb[i].unsqueeze(0),
                "x2": x_src[i].unsqueeze(0), 'x2_ls': x_len[i].unsqueeze(0),
                'x2_p': x_src[i].unsqueeze(0), 'x2_p_ls': x_len[i].unsqueeze(0),
                "y2": y_rgb[i].unsqueeze(0),
                'xt': x_tgt,
            }
            z = self.q.sample(eval_args_xy, sample_shape=[sample_size])['z']
            eval_args_xyz = {'z': z}
            eval_args_xyz.update(eval_args_xy)
            p_probs = self.prior.log_prob().eval({'z': z})
            p_x_z_probs = self.p_x.log_prob().eval(eval_args_xyz)
            p_y_z_probs = self.p_y.log_prob().eval(eval_args_xyz)

            a = self.q.get_params(eval_args_xy)
            expert = Normal(loc=a['loc'], scale=a['scale'],
                            var=["z"], features_shape=[1, self.z_dim], name="expert").to(device)
            q_probs = expert.log_prob().eval(eval_args_xyz)

            a = logsumexp((p_x_z_probs + p_y_z_probs + p_probs - q_probs).detach().cpu().numpy(), axis=-1)
            result.append(float(a))
        return result

    def model_args_colors(self, y_rgb, x_src, x_tgt, x_len, y_rgb_u, x_src_u, x_tgt_u, x_len_u, beta=1.0, is_supervised=1.0):
        return {
            "x1": x_src, 'x1_ls': x_len,
            "x1_u": x_src_u, 'x1_ls_u': x_len_u,
            "y1": y_rgb,

            'x1_p': x_src, 'x1_p_ls': x_len,
            'x1_p_u': x_src_u, 'x1_p_u_ls': x_len_u,
            "y1_u": y_rgb_u,

            "x2": x_src, 'x2_ls': x_len,
            "x2_u": x_src_u, 'x2_ls_u': x_len_u,
            "y2": y_rgb,

            'x2_p': x_src, 'x2_p_ls': x_len,
            'x2_p_u': x_src_u, 'x2_p_u_ls': x_len_u,
            "y2_u": y_rgb_u,

            'xt': x_tgt,
            'xt_u': x_tgt_u,

            "beta": beta,
            'is_supervised': is_supervised,

        }

    def eval_args_colors(self, y_rgb, x_src, x_tgt, x_len):
        return {
            "y1": y_rgb,
            "x1": x_src, 'x1_ls': x_len,
            "x1_u": x_src, 'x1_ls_u': x_len,
            "y1_u": y_rgb,
            'x1_p': x_src, 'x1_p_ls': x_len,
            'x1_p_u': x_src, 'x1_p_u_ls': x_len,

            "y2": y_rgb,
            "x2": x_src, 'x2_ls': x_len,
            "x2_u": x_src, 'x2_ls_u': x_len,
            "y2_u": y_rgb,
            'x2_p': x_src, 'x2_p_ls': x_len,
            'x2_p_u': x_src, 'x2_p_u_ls': x_len,

            'xt': x_tgt,
            'xt_u': x_tgt,
        }

    def sample_z(self, y, sample_shape=None):
        if sample_shape:
            return self.q2.sample({"y1": y, "y2": y}, sample_shape=[sample_shape], return_all=False)
        return self.q2.sample({"y1": y, "y2": y}, return_all=False)

    def sample_z_from_x(self, x, sample_shape=None):
        if sample_shape:
            return self.q1.sample({"x1": x, "x2": x}, sample_shape=[sample_shape], return_all=False)
        return self.q1.sample({"x1": x, "x2": x}, return_all=False)

    def sample_z_all(self, x, y, sample_shape=None):
        if sample_shape:
            return self.q.sample({"x1": x, "x2": x, "y1": y, "y2": y}, sample_shape=[sample_shape], return_all=False)
        return self.q.sample({"x1": x, "x2": x, "y1": y, "y2": y}, return_all=False)

    def sample_z_from_x_graph(self, x, g, sample_shape=None):
        if sample_shape:
            return self.q1.sample({"x1": x, "x2": x, "graph1": g, "graph2": g}, sample_shape=[sample_shape], return_all=False)
        return self.q1.sample({"x1": x, "x2": x, "graph1": g, "graph2": g}, return_all=False)

    def sample_z_all_graph(self, x, y, g, sample_shape=None):
        if sample_shape:
            return self.q.sample({"x1": x, "x2": x, "y1": y, "y2": y, "graph1": g, "graph2": g}, sample_shape=[sample_shape], return_all=False)
        return self.q.sample({"x1": x, "x2": x, "y1": y, "y2": y, "graph1": g, "graph2": g}, return_all=False)

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
        for m in [self.q1_x, self.q2_y, self.q_star_1_y, self.q_star_2_x]:
            result += sum(p.numel() for p in m.parameters() if p.requires_grad)
        return result