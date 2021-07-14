import matplotlib
matplotlib.use('Agg')
import torch
import torch.utils.data
import numpy as np
import pandas as pd
import os
import sys
import shutil

from models.MVAE import MVAE
from models.JMVAE import JMVAE
from models.SpectraVAE import SpectraVAE
from models.SpectraVAE_terms import SpectraVAE as SpectraVAE_terms
from models.SpectraVAE_shared import SpectraVAEShared
from torch.utils.data.sampler import SubsetRandomSampler

from experiments.spectra_fingerprints_paper.spectra_fingerprints_paper_inference import GeneratorX, GeneratorY, \
    InferenceX, InferenceX_missing, InferenceY, InferenceY_missing, z_dim, \
    InferenceJoint

batch_size = 250
epochs = 301
annealing_epochs = 0
best_loss = sys.maxsize

if sys.argv[3].startswith('cuda') and torch.cuda.is_available():
    device = sys.argv[3]
else:
    device = "cpu"

print('Device', device)

kwargs = {'batch_size': batch_size, 'num_workers': 10, 'pin_memory': True}

root = "."


BASE_DIR = '/home/svegal'


def align_spectra(resolution, max_mz, row):
    numbers = np.arange(0, max_mz, step=resolution)
    result = [0]*len(numbers)
    for i in row:
        idx = np.searchsorted(numbers, i[0])
        try:
            result[idx] += i[1]
            # result[idx] = max(i[1], result[idx])
        except IndexError:
            result[-1] = i[1]
    return np.array(result)


def parse_spectrum(m):
    spec = align_spectra(0.1, 1100, m)
    # spec = align_spectra(0.05, 2500, m)
    spec = np.log(1. + spec)
    return spec


def save_checkpoint(model_obj, is_best, model_name, folder=f'{BASE_DIR}/saved_models', filename='checkpoint_{}.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(dict(distributions=model_obj.model.distributions.state_dict()), os.path.join(folder, filename.format(model_name)))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename.format(model_name)),
                        os.path.join(folder, 'model_best_{}.pth.tar'.format(model_name)))


def load_checkpoint(model_obj, model_name, is_best=True):
    if is_best:
        filename = 'model_best_{}.pth.tar'
    else:
        filename = 'checkpoint_{}.pth.tar'
    file_path = os.path.join(f'{BASE_DIR}/saved_models', filename.format(model_name))
    checkpoint = torch.load(file_path)
    model_obj.model.distributions.load_state_dict(checkpoint['distributions'])
    model_obj.model.distributions.eval()


def parse_spectrum_string(row):
    string = row['spectrum']
    return np.array([(float(s.split(':')[0]), float(s.split(':')[1])) for s in string.split()])


def save_reconstruction(model_obj, data_loader, n_share, suffix=''):
    print('Reconstruction')

    def save_cpu(f, v):
        np.save(f'{BASE_DIR}/reconstructions/{model_obj.name}_{n_share}_{keyword}{suffix}_{f}.npy', v.detach().to("cpu").numpy())

    def save_cpu_spectra(f, v):
        v = v.detach().to("cpu").numpy()
        np.save(f'{BASE_DIR}/reconstructions/{model_obj.name}_{n_share}_{keyword}{suffix}_{f}.npy', v)

    for x, y in data_loader:
        up_all, down_all = x.to(device), y.to(device).float()
        y = y.to(device).float().detach()

        z_up = model_obj.sample_z_from_x(up_all)
        recon_up_up = model_obj.reconstruct_x(z_up)
        recon_up_down = model_obj.reconstruct_y(z_up)

        z_down = model_obj.sample_z(down_all)
        recon_down_up = model_obj.reconstruct_x(z_down)
        recon_down_down = model_obj.reconstruct_y(z_down)

        z_full = model_obj.sample_z_all(up_all, down_all)
        recon_full_up = model_obj.reconstruct_x(z_full)
        recon_full_down = model_obj.reconstruct_y(z_full)

        save_cpu_spectra('true_spectrum', up_all)
        save_cpu('true_fp', down_all)
        save_cpu_spectra('spectrum_from_spectrum', recon_up_up)
        save_cpu('fp_from_spectrum', recon_up_down)
        save_cpu_spectra('spectrum_from_fp', recon_down_up)
        save_cpu('fp_from_fp', recon_down_down)
        save_cpu_spectra('spectrum_from_all', recon_full_up)
        save_cpu('fp_from_all', recon_full_down)
        save_cpu('latent_spectrum', z_up['z'])
        save_cpu('latent_fp', z_down['z'])
        save_cpu('latent_all', z_full['z'])
        return


def save_reconstruction_spectrum(model_obj, data_loader, n_share, suffix=''):
    print('Reconstruction')

    def save_cpu(f, v):
        np.save(f'{BASE_DIR}/reconstructions/{model_obj.name}_{n_share}_{keyword}{suffix}_{f}.npy', v.detach().to("cpu").numpy())

    def save_cpu_spectra(f, v):
        v = v.detach().to("cpu").numpy()
        np.save(f'{BASE_DIR}/reconstructions/{model_obj.name}_{n_share}_{keyword}{suffix}_{f}.npy', v)

    for x, y in data_loader:
        up_all, down_all = x.to(device), y.to(device).float()

        z_up = model_obj.sample_z_from_x(up_all)
        recon_up_up = model_obj.reconstruct_x(z_up)
        recon_up_down = model_obj.reconstruct_y(z_up)

        z_down = model_obj.sample_z(down_all)
        recon_down_up = model_obj.reconstruct_x(z_down)
        recon_down_down = model_obj.reconstruct_y(z_down)

        save_cpu_spectra('true_spectrum', up_all)
        save_cpu('true_fp', down_all)
        save_cpu_spectra('spectrum_from_spectrum', recon_up_up)
        save_cpu('fp_from_spectrum', recon_up_down)

        save_cpu_spectra('spectrum_from_fp', recon_down_up)
        save_cpu('fp_from_fp', recon_down_down)
        return


def evaluate_accuracy(model_obj, data_loader):
    correct_full, correct_up, correct_down = [], [], []
    criterion = torch.nn.BCELoss()
    print('accuracy')
    for x, y in data_loader:
        up_all, down_all = x.to(device), y.to(device).float()
        y = y.to(device).float().detach()

        z_up = model_obj.sample_z_from_x(up_all)
        recon_up_up = model_obj.reconstruct_x(z_up)
        recon_up_down = model_obj.reconstruct_y(z_up)

        z_down = model_obj.sample_z(down_all)
        recon_down_up = model_obj.reconstruct_x(z_down)
        recon_down_down = model_obj.reconstruct_y(z_down)

        z_full = model_obj.sample_z_all(up_all, down_all)
        recon_full_up = model_obj.reconstruct_x(z_full)
        recon_full_down = model_obj.reconstruct_y(z_full)

        print('recon_full_down', recon_full_down)
        print('recon_up_down', recon_up_down)
        print('recon_down_down', recon_down_down)
        print('y', y)

        correct_full.append(criterion(recon_full_down, y).item())
        correct_up.append(criterion(recon_up_down, y).item())
        correct_down.append(criterion(recon_down_down, y).item())

    return np.mean(correct_full), np.mean(correct_up), np.mean(correct_down)


def get_beta(epoch, i):
    if epoch < annealing_epochs:
        N_mini_batches = int(N_data / batch_size)
        return float(i + (epoch - 1) * N_mini_batches + 1) / float(annealing_epochs * N_mini_batches)
    else:
        return 1.0


current_beta = 0


def run_semisupervised(model_obj, no_labels):
    global best_loss, current_beta

    betas = []
    model = model_obj.model

    def train(epoch):
        global current_beta
        train_loss = 0
        train_loss_only_x = 0
        train_loss_only_y = 0
        train_xy_iterator = train_loader_supervised.__iter__()
        train_x_iterator = train_loader_unsupervised_x.__iter__()
        train_y_iterator = train_loader_unsupervised_y.__iter__()
        bsize = train_loader_unsupervised_y.batch_size
        dsize = train_loader_unsupervised_y.dataset
        for i in range(len(train_loader_unsupervised_y)):
            print('Iteration', i, 'out of', len(train_loader_unsupervised_y), flush=True)

            try:
                x, y = next(train_xy_iterator)
            except StopIteration:
                train_xy_iterator = train_loader_supervised.__iter__()
                x, y = next(train_xy_iterator)

            try:
                x_u, _ = next(train_x_iterator)
            except StopIteration:
                train_x_iterator = train_loader_unsupervised_x.__iter__()
                x_u, _ = next(train_x_iterator)

            try:
                _, y_u = next(train_y_iterator)
            except StopIteration:
                train_y_iterator = train_loader_unsupervised_y.__iter__()
                _, y_u = next(train_y_iterator)

            beta = get_beta(epoch, i)
            current_beta = beta
            x, y = x.to(device), y.to(device).float()
            x_u = x_u.to(device)
            current_beta = beta
            y_u = y_u.to(device).float()
            loss = model.train(model_obj.model_args(
                x,
                y,
                x_u,
                y_u,
                beta=beta
            ))
            torch.cuda.empty_cache()
            print('Loss', float(loss), flush=True)
            train_loss += float(loss)
        train_loss = train_loss * bsize / len(dsize)
        train_loss_only_x = train_loss_only_x * bsize / len(dsize)
        train_loss_only_y = train_loss_only_y * bsize / len(dsize)
        return train_loss, train_loss_only_x, train_loss_only_y, current_beta

    def test(epoch):
        test_loss = 0
        result = []
        test_xy_iterator = test_loader_supervised.__iter__()
        test_x_iterator = test_loader_unsupervised_x.__iter__()
        test_y_iterator = test_loader_unsupervised_y.__iter__()
        bsize = test_loader_unsupervised_y.batch_size
        dsize = test_loader_unsupervised_y.dataset
        print('test len', len(test_loader_unsupervised_y), flush=True)
        for i in range(len(test_loader_unsupervised_y)):

            try:
                x, y = next(test_xy_iterator)
            except StopIteration:
                test_xy_iterator = test_loader_supervised.__iter__()
                x, y = next(test_xy_iterator)

            try:
                x_u, _ = next(test_x_iterator)
            except StopIteration:
                test_x_iterator = test_loader_unsupervised_x.__iter__()
                x_u, _ = next(test_x_iterator)

            try:
                _, y_u = next(test_y_iterator)
            except StopIteration:
                test_y_iterator = test_loader_unsupervised_y.__iter__()
                _, y_u = next(test_y_iterator)

            x, y = x.to(device), y.to(device).float()
            x_u = x_u.to(device)
            y_u = y_u.to(device).float()
            # print('Iteration', i, flush=True)
            loss = model.test(model_obj.model_args(
                x,
                y,
                x_u,
                y_u
            ))
            test_loss += float(loss)
            result.extend([0] * x.shape[0])
        epoch_loss = test_loss * bsize / len(dsize)
        result = np.array(result)
        return epoch_loss, result

    train_losses = []
    train_losses_only_x = []
    train_losses_only_y = []
    test_losses = []
    likelihood_means = []
    likelihood_stds = []

    accuracies_full_test = []
    accuracies_up_test = []
    accuracies_down_test = []

    accuracies_full_train = []
    accuracies_up_train = []
    accuracies_down_train = []

    for epoch in range(epochs):
        print('Model', model_obj.name, ', epoch', epoch, ', no labels', no_labels)
        loss, loss_only_x, loss_only_y, beta = train(epoch)
        train_losses.append(loss)
        train_losses_only_x.append(loss_only_x)
        train_losses_only_y.append(loss_only_y)
        betas.append(beta)
        t_l, lik = test(epoch)
        test_losses.append(t_l)
        print(lik)
        ac_full_test, ac_up_test, ac_down_test = evaluate_accuracy(model_obj, test_loader_supervised)
        ac_full_train, ac_up_train, ac_down_train = evaluate_accuracy(model_obj, train_loader_train_ac)

        is_best = ac_up_test < best_loss
        best_loss = min(ac_up_test, best_loss)
        # save_checkpoint(model_obj, is_best, '{}_{}_{}'.format(model_obj.name, no_labels, keyword))
        if epoch % 10 == 0:
            save_checkpoint(model_obj, False, f'{model_obj.name}_{no_labels}_{keyword}_{epoch}')

        accuracies_full_test.append(ac_full_test)
        accuracies_up_test.append(ac_up_test)
        accuracies_down_test.append(ac_down_test)

        accuracies_full_train.append(ac_full_train)
        accuracies_up_train.append(ac_up_train)
        accuracies_down_train.append(ac_down_train)

        likelihood_means.append(lik.mean())
        likelihood_stds.append(lik.std())
        result = pd.DataFrame({
            'train_loss': train_losses,
            'train_loss_only_x': train_losses_only_x,
            'train_loss_only_y': train_losses_only_y,
            'likelihood_mean': likelihood_means,
            'likelihood_std': likelihood_stds,
            'test_loss': test_losses,
            'last_betas': betas,
            'n_parameters': [model_obj.get_number_of_parameters()] * len(train_losses),

            'loss_full_test': accuracies_full_test,
            'loss_spectra_test': accuracies_up_test,
            'loss_fp_test': accuracies_down_test,

            'loss_full_train': accuracies_full_train,
            'loss_spectra_train': accuracies_up_train,
            'loss_fp_train': accuracies_down_train,
        })
        result.to_csv('{}_{}_{}.csv'.format(model_obj.name, no_labels, keyword), index=None)

    result = pd.DataFrame({
        'train_loss': train_losses,
        'train_loss_only_x': train_losses_only_x,
        'train_loss_only_y': train_losses_only_y,
        'test_loss': test_losses,
        'likelihood_mean': likelihood_means,
        'likelihood_std': likelihood_stds,
        'last_betas': betas,
        'n_parameters': [model_obj.get_number_of_parameters()]*len(train_losses),

        'loss_full_test': accuracies_full_test,
        'loss_spectra_test': accuracies_up_test,
        'loss_fp_test': accuracies_down_test,

        'loss_full_train': accuracies_full_train,
        'loss_spectra_train': accuracies_up_train,
        'loss_fp_train': accuracies_down_train,
    })
    result.to_csv('{}_{}_{}.csv'.format(model_obj.name, no_labels, keyword), index=None)


keyword = 'paper_unsupervised_fp_unmerged'
# keyword = 'paper_unsupervised_casmi'
# keyword = 'paper_unsupervised_fp_1e4'
# keyword = 'paper_unsupervised_fp_unmerged_unsup'


class CSVFileDataset(torch.utils.data.dataset.Dataset):
    """CASMI2017 dataset"""

    def __init__(self, X, y, indices=None):
        self.X = X
        self.y = y
        self.indices = indices

    def __len__(self):
        return len(self.X)

    def parse_spectrum_in(self, x):
        fs = []
        for pair in x.split(' '):
            i, j = pair.split(':')
            fs.append((float(i), float(j)))
        return np.array(parse_spectrum(fs)).astype(np.float32)

    def __getitem__(self, idx):
        return self.parse_spectrum_in(self.X[idx]), np.array(list(map(int, self.y[idx])))[self.indices]


class FPDataset(torch.utils.data.dataset.Dataset):
    """Spectra dataset"""

    def __init__(self, y, y_func=lambda x: x):
        self.X = None
        self.y = y
        self.y_func = y_func

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return [0], self.y_func(self.y[idx])


def load_smiles_to_fp(path):
    smiles_to_fingerprints = {}
    with open(path) as f:
        for line in f:
            smiles, fp = line.strip().split(',')
            smiles_to_fingerprints[smiles] = fp
    return smiles_to_fingerprints


def parse_fp_orbitrap(fp):
    return np.array([int(i) for i in fp])


def get_samplers(N_data):
    indices = list(range(N_data))
    np.random.shuffle(indices)
    valid_idx_x = copy.copy(indices)
    np.random.shuffle(valid_idx_x)
    valid_idx_y = copy.copy(indices)
    np.random.shuffle(valid_idx_y)
    unsupervised_sampler_x = SubsetRandomSampler(valid_idx_x)
    unsupervised_sampler_y = SubsetRandomSampler(valid_idx_y)
    supervised_sampler = SubsetRandomSampler(indices)
    return unsupervised_sampler_x, unsupervised_sampler_y, supervised_sampler


ap = argparse.ArgumentParser()

ap.add_argument("-t", "--train", required=True, help="Train set, a csv file with comma as delimiter, required columns 'spectrum' and 'fp'")
ap.add_argument("-v", "--test", required=True, help="Test set, a csv file with comma as delimiter, required columns 'spectrum' and 'fp'")
ap.add_argument("-u", "--train_molecules", required=False, help="Unsupervised fingerprint train set. If not provided, the algorithm runs in supervised mode")
ap.add_argument("-m", "--model", required=True, help="Model name (SVAE or JMVAE). JMVAE only runs in supervised mode even if train_molecules provided")
ap.add_argument("-k", "--keyword", required=False, default='fingerprints', help="Additional keyword for the trained model name. Default 'fingerprints'")
ap.add_argument("-l", "--load", required=False, help="Trained model name to evaluate or continue the training")
ap.add_argument("-d", "--device", required=False, default='cpu', help="Device name, default cpu")
ap.add_argument("-e", "--eval", action="store_true", help="If eval flag is set, model predicts spectra from molecules for 1000 samples from the test set and predicts molecules from spectra")
args = vars(ap.parse_args())

keyword = args['keyword']
print('Keyword', keyword)

df_metid_train = pd.read_csv(args['train'])
df_metid_test = pd.read_csv(args['test'])
if args['train_molecules']:
    unsupervised_fp = list(pd.read_csv(args['train_molecules'])['fp'])
    is_sup = False
else:
    is_sup = True

indices_train_ac = np.random.choice(len(df_metid_train), len(df_metid_test))
df_metid_train_ac = df_metid_train.iloc[indices_train_ac]
df_metid_train_ac = df_metid_train_ac.reset_index()

N_data = len(df_metid_train)
unsupervised_sampler_x, unsupervised_sampler_y, supervised_sampler = get_samplers(N_data)

device = args['device']

train_loader_supervised = torch.utils.data.DataLoader(
    CSVFileDataset(df_metid_train['spectrum'], df_metid_train['fp']),
    sampler=supervised_sampler, **kwargs
)
if is_sup or args['model'] == 'JMVAE':
    train_loader_unsupervised_x = torch.utils.data.DataLoader(
        CSVFileDataset(df_metid_train['spectrum'], df_metid_train['fp']),
        sampler=unsupervised_sampler_x, **kwargs
    )
    train_loader_unsupervised_y = torch.utils.data.DataLoader(
        CSVFileDataset(df_metid_train['spectrum'], df_metid_train['fp']),
        sampler=unsupervised_sampler_y, **kwargs
    )
else:
    train_loader_unsupervised_x = torch.utils.data.DataLoader(
        CSVFileDataset(
            list(df_metid_train['spectrum']),
            list(df_metid_train['fp']),
        ),
        shuffle=True, **kwargs
    )
    train_loader_unsupervised_y = torch.utils.data.DataLoader(
        CSVFileDataset(
            list(df_metid_train['spectrum'])[0]*len(unsupervised_fp),
            unsupervised_fp,
        ),
        shuffle=True, **kwargs
    )

test_loader_supervised = torch.utils.data.DataLoader(
    CSVFileDataset(df_metid_test['spectrum'], df_metid_test['fp'],),
    shuffle=False, **kwargs
)
test_loader_unsupervised_x = torch.utils.data.DataLoader(
    CSVFileDataset(df_metid_test['spectrum'], df_metid_test['fp']),
    shuffle=False, **kwargs
)
test_loader_unsupervised_y = torch.utils.data.DataLoader(
    CSVFileDataset(df_metid_test['spectrum'], df_metid_test['fp']),
    shuffle=False, **kwargs
)

train_loader_train_ac = torch.utils.data.DataLoader(
    CSVFileDataset(df_metid_train_ac['spectrum'], df_metid_train_ac['fp_short']),
    shuffle=False, **kwargs
)


models_classes = {
    'SVAE': SpectraVAE_terms,
    'JMVAE': JMVAE,
}

model_class = models_classes[args['model']]

def init_model(model_class):
    p_x = GeneratorX()
    p_y = GeneratorY()

    q_x = InferenceX()
    q_y = InferenceY()

    q_star_y = InferenceY_missing()
    q_star_x = InferenceX_missing()

    q = InferenceJoint()

    return model_class(z_dim, {"lr": 1e-4}, q_x, q_y, p_x, p_y, q=q, q_star_y=q_star_y, q_star_x=q_star_x, device=device)


model_obj = init_model(model_class)

no_labels = int(is_sup)
print('No labels', no_labels)

if args['eval']:
    name = args['load']
    load_checkpoint(model_obj, name, is_best=False)

    test_loader_supervised = torch.utils.data.DataLoader(
        CSVFileDataset(df_metid_test['spectrum'], df_metid_test['fp']),
        shuffle=False, batch_size=1000
    )
    train_loader_train_ac = torch.utils.data.DataLoader(
        CSVFileDataset(df_metid_train_ac['spectrum'], df_metid_train_ac['fp']),
        shuffle=False, batch_size=1000
    )
    save_reconstruction_spectrum(model_obj, test_loader_supervised, n_share, suffix='_test')
    save_reconstruction_spectrum(model_obj, train_loader_train_ac, n_share, suffix='_train')
else:
    print('Started learning')
    if args['load']:
        name = args['load']
        load_checkpoint(model_obj, name, is_best=False)
    run_semisupervised(model_obj, no_labels)

