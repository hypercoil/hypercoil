# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Generative model (extremely rough WIP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Code largely adapted from here:
https://github.com/fmu2/Wasserstein-BiGAN/blob/master/util.py

Original code by Yiwu Zhong and Fangzhou Mu.

Reuse / modification as permitted under the MIT license.

MIT License

Copyright (c) 2019 Fangzhou Mu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os
import torch
import pathlib
import pandas as pd
import webdataset as wds
import matplotlib.pyplot as plt
from torch.nn import Module, Parameter
from torch.autograd import grad
from functools import partial
from itertools import chain

from hypercoil.data.collate import gen_collate, extend_and_bind
from hypercoil.engine import Epochs
from hypercoil.functional import corr, sym2vec
from hypercoil.loss import UnilateralNormedLoss, ModularityLoss
from hypercoil.nn.activation import CorrelationNorm
from hypercoil.nn.sylo import SyloResNet, Sylo
from hypercoil.nn.recombinator import QueryEncoder, Recombinator
from hypercoil.nn.window import WindowAmplifier

import matplotlib.pyplot as plt


dtype = torch.float
device = 'cpu'

n_subjects = 10
n_tasks = 6

data_dir = '/mnt/andromeda/Data/MSC_schaefer/{000000..000781}.tar'
results = '/tmp/generative/'

# architectural hyperparameters
channel_sequence = (1, 12, 36, 60)
generator_channel_sequence = (24, 12, 12, 32)
block_sequence = (1, 1, 1)
dim_sequence = (400, 100, 25, 10)
lattice_order_sequence = (4, 2, 1)
block_length = 2
channel_multiplier = 2
embedding_dim = 5 # dim per embedding
noise_dim = 10

# training hyperparameters
lr = 0.002
window_size = 200
max_epoch = 5000
batch_size = 10
critic_steps_per_generator_step = 3
positive_only_nu = 0.1
community_nu = 0.01
log_interval = 1
log_interval = log_interval * (1 + critic_steps_per_generator_step)


class Critic(Module):
    def __init__(self):
        super().__init__()
        self.model = SyloResNet(
            in_dim=dim_sequence[0],
            in_channels=channel_sequence[0],
            community_dim=8,
            dim_sequence=dim_sequence[1:],
            channel_sequence=channel_sequence[1:],
            block_sequence=block_sequence,
            lattice_order_sequence=lattice_order_sequence,
            n_lattices=1,
            channel_multiplier=channel_multiplier,
            norm_layer=torch.nn.InstanceNorm2d,
            nlin=partial(torch.nn.LeakyReLU, negative_slope=0.1) #None #CorrelationNorm
        )
        final_dim = dim_sequence[-1]
        d0, d1 = channel_sequence[-1], final_dim * (final_dim - 1) // 2
        #TODO: use real init . . .
        W_L = torch.randn(d0, 1, dtype=dtype, device=device) / d0
        W_R = torch.randn(d1, 1, dtype=dtype, device=device) / d1
        self.W_L = Parameter(W_L)
        self.W_R = Parameter(W_R)

    def set_potentials(self, potentials):
        self.model.set_potentials(potentials)

    def forward(self, x, z):
        y = self.model(x.unsqueeze(1), query=z)
        return self.W_L.t() @ sym2vec(y) @ self.W_R


class Encoder(Module):
    def __init__(self):
        super().__init__()
        self.model = SyloResNet(
            in_dim=dim_sequence[0],
            in_channels=channel_sequence[0],
            community_dim=8,
            dim_sequence=dim_sequence[1:],
            channel_sequence=channel_sequence[1:],
            block_sequence=block_sequence,
            lattice_order_sequence=lattice_order_sequence,
            n_lattices=1,
            channel_multiplier=channel_multiplier,
            norm_layer=torch.nn.InstanceNorm2d,
            nlin=partial(torch.nn.LeakyReLU, negative_slope=0.1) #None #CorrelationNorm
        )
        final_dim = dim_sequence[-1]
        d0, d1 = channel_sequence[-1], final_dim * (final_dim - 1) // 2
        W_L = torch.randn(d0, 5, dtype=dtype, device=device) / d0
        W_R = torch.randn(d1, 4, dtype=dtype, device=device) / d1
        self.W_L = Parameter(W_L)
        self.W_R = Parameter(W_R)

    def set_potentials(self, critic):
        for i, layer in enumerate(self.model.model.layers):
            c_E = layer[0].compression
            c_C = critic.model.model.layers[i][0].compression
            with torch.no_grad():
                c_E.C[:] = c_C.C.clone()
                c_E.mask[:] = c_C.mask.clone()
                c_E.sign[:] = c_C.sign.clone()
                c_E.sparsity = c_C.sparsity
                c_E.initialised = True

    def forward(self, x, query=None):
        batch_size = x.size(0)
        z_hat = self.model(x.unsqueeze(1), query=query)
        z_hat = self.W_L.t() @ sym2vec(z_hat) @ self.W_R
        return z_hat.view(batch_size, -1)


class Generator(Module):
    def __init__(self):
        super().__init__()
        self.model = SyloResNet(
            in_dim=dim_sequence[-1],
            in_channels=generator_channel_sequence[0],
            community_dim=0,
            dim_sequence=dim_sequence[-2::-1],
            channel_sequence=generator_channel_sequence[1:],
            block_sequence=block_sequence,
            lattice_order_sequence=lattice_order_sequence[::-1],
            n_lattices=1,
            channel_multiplier=channel_multiplier,
            norm_layer=torch.nn.InstanceNorm2d,
            nlin=partial(torch.nn.LeakyReLU, negative_slope=0.1) #None #CorrelationNorm
        )
        self.final = Recombinator(
            in_channels=generator_channel_sequence[-1],
            out_channels=1
        )

    def set_potentials(self, critic):
        for i, layer in enumerate(critic.model.model.layers):
            c_G = self.model.model.layers[-(i + 1)][0].compression
            c_C = layer[0].compression
            with torch.no_grad():
                c_G.C[:] = c_C.C.clone().transpose(-1, -2)
                c_G.mask[:] = c_C.mask.clone().transpose(-1, -2)
                c_G.sign[:] = c_C.sign.clone()
                c_G.sparsity = c_C.sparsity
                c_G.initialised = True

    def forward(self, z, query=None):
        batch_size = z.size(0)
        in_dim = dim_sequence[-1]
        #TODO: no reason at all this view will work in the general case.
        z = z.view(batch_size, in_dim, -1)
        z = z @ z.transpose(-1, -2)
        x_tilde = self.model(z.view(batch_size, 1, in_dim, in_dim), query=query)
        return self.final(x_tilde).squeeze()


class QueryNet(Module):
    def __init__(self):
        super().__init__()
        channels = (
            list(channel_sequence)[1:] + list(generator_channel_sequence)[1:]
        )
        blocks = list(block_sequence) * 2
        query_dim = [[c for _ in range(b * block_length)] for b, c in
                     zip(blocks, channels)]
        query_dim = list(chain(*query_dim))
        self.model = QueryEncoder(
            num_embeddings=(n_subjects, n_tasks), # subject and task
            embedding_dim=embedding_dim,
            noise_dim=noise_dim,
            query_dim=query_dim,
            common_layer_dim=(16, 16),
            specific_layer_dim=(16,)
        )

    def encode(self, embedding):
        nrecomb = sum([b * block_length for b in block_sequence])
        query, _ = self.model(embedding, skip_embedding=True)
        query_enc, query_gen =  query[:nrecomb], query[nrecomb:]
        return query_enc, query_gen

    def forward(self, subject_query, task_query):
        nrecomb = sum([b * block_length for b in block_sequence])
        query, embedding = self.model(x=[subject_query, task_query])
        query_enc, query_gen =  query[:nrecomb], query[nrecomb:]
        return (query_enc, query_gen), embedding


class QueryBiGAN(Module):
    """
    Based on the Wasserstein ALI implementation here:
    https://github.com/fmu2/Wasserstein-BiGAN/blob/master/util.py
    with modifications to make a conditional BiGAN.

    Original code by Yiwu Zhong and Fangzhou Mu.

    Reuse / modification as permitted under the MIT license.

    MIT License

    Copyright (c) 2019 Fangzhou Mu

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    def __init__(self, encoder, generator, critic, qnet, gradient_nu=10):
        super().__init__()
        self.encoder = encoder
        self.generator = generator
        self.critic = critic
        self.qnet = qnet
        self.gradient_nu = gradient_nu

    def encode(self, x, query=None):
        return self.encoder(x, query=query)

    def generate(self, z, query=None):
        return self.generator(z, query=query)

    def critique(self, x, z_hat, x_tilde, z):
        data_size = x.size(0)
        X = torch.cat((x, x_tilde), 0)
        Z = torch.cat((z_hat, z), 0)
        out = self.critic(X, Z)
        return out[:data_size], out[data_size:]

    def reconstruct(self, x, query_E=None, query_G=None):
        return self.generate(self.encode(x, query=query_E), query=query_G)

    def _grad_penalty_interpolate(self, n, x, z_hat, x_tilde, z):
        """
        Interpolate between data and synthetic examples as a
        convex combination.
        """
        broadcast_conform = [1 for _ in range(x.dim() - 1)]
        eps = torch.rand(n, *broadcast_conform, device=x.device)
        interpolate_x = eps * x + (1 - eps) * x_tilde
        broadcast_conform = [1 for _ in range(z.dim() - 1)]
        eps = torch.rand(n, *broadcast_conform, device=z.device)
        interpolate_z = eps * z_hat + (1 - eps) * z
        interpolate_x.requires_grad = True
        interpolate_z.requires_grad = True
        return interpolate_x, interpolate_z

    def gradient_penalty(self, x, z_hat, x_tilde, z):
        """
        Encourage the gradient for each example to go toward 1 to
        improve conformance with the Lipschitz constraint.

        Gradient penalty for Wasserstein GANs, as described here:
        https://arxiv.org/pdf/1704.00028.pdf
        and following closely the implementation by Yiwu Zhong and
        Fangzhou Mu for BiGANs.
        """
        #TODO: Better replace this with spectral norm when we're not
        # so time-constrained.
        batch_size = x.size(0)
        interpolate_x, interpolate_z = self._grad_penalty_interpolate(
            n=batch_size, x=x, z_hat=z_hat, x_tilde=x_tilde, z=z
        )
        #print(x.shape, z.shape, x_tilde.shape, z_hat.shape)
        #print(interpolate_x.shape, interpolate_z.shape)
        if self.qnet is not None:
            interpolate_z_enc, _ = self.qnet.encode(interpolate_z)
            critic_scalar = self.critic(interpolate_x, interpolate_z_enc).sum()
        else:
            critic_scalar = self.critic(interpolate_x, interpolate_z).sum()
        grad_x, grad_z = grad(
            critic_scalar, (interpolate_x, interpolate_z),
            retain_graph=True, create_graph=True
        )
        grads = torch.cat((
            grad_x.view(batch_size, -1),
            grad_z.view(batch_size, -1)
        ), dim=-1)
        #print(grads)
        return ((grads.norm(2, dim=-1) - 1) ** 2).mean().abs()

    def forward(self, x, z, query_E=None, query_G=None):
        z_hat = self.encode(x, query=query_E)
        x_tilde = self.generate(z, query=query_G)
        y_data, y_synthetic = self.critique(
            x=x, z_hat=z_hat, x_tilde=x_tilde, z=z
        )
        EG_loss = (y_data - y_synthetic).mean()
        GP_loss = self.gradient_nu * self.gradient_penalty(
            x=x.data,
            z_hat=z_hat.data,
            x_tilde=x_tilde.data,
            z=z.data
        )
        #print(GP_loss)
        C_loss = -EG_loss + GP_loss
        return C_loss, EG_loss, (x, z_hat), (x_tilde, z)


class ConnectomeQueryBiGAN(QueryBiGAN):
    def set_potentials(self, potentials):
        self.critic.set_potentials(potentials)
        self.encoder.set_potentials(self.critic)
        self.generator.set_potentials(self.critic)

    def critique(self, x, z_hat, x_tilde, z):
        data_size = x.size(0)
        X = torch.cat((x, x_tilde), 0)
        Z = torch.cat((z_hat, z), 0)
        Z, _ = self.qnet.encode(Z)
        out = self.critic(X, Z)
        return out[:data_size], out[data_size:]

    def forward(self, x, subject_query, task_query):
        (q_E, q_G), z = self.qnet(
            subject_query=subject_query,
            task_query=task_query
        )
        return super().forward(
            x=x, z=z, query_E=q_E, query_G=q_G
        )


def correlate(data):
    ts = data.to(device=device)
    nanmask = torch.isnan(ts)
    ts[nanmask] = 0
    nanmask = nanmask.sum(-2).to(dtype=torch.bool)
    return corr(ts, weight=~nanmask)


def regularisation(network, x=None):
    nonnegative_loss = 0
    for m in network.modules():
        if isinstance(m, Sylo):
            nonnegative_loss += positive_only(-m.weight_L)
    if x is None:
        modularity_loss = 0
    else:
        C = torch.softmax(
            network.sylo1['community'].weight_L.squeeze().t(), -1
        )
        modularity_loss = community_loss(A=x, C=C)
    return nonnegative_loss, modularity_loss


def wndw_map(dict, keys, window):
    try:
        return {
            k : (window(v)[0] if k in keys else v)
            for k, v in dict.items()
        }
    except ValueError:
        return dict


def unsqueeze(dict, keys):
    return {k : v.unsqueeze(0) if k in keys else v for k, v in dict.items()}

def remove(dict):
    del dict['__key__']
    return dict

os.makedirs(results, exist_ok=True)
collate = partial(gen_collate, concat=extend_and_bind)
window = WindowAmplifier(window_size=window_size)
wndw_map = partial(wndw_map, keys='bold', window=window)
map2 = partial(unsqueeze, keys=('task', 'sub'))
dpl = wds.DataPipeline(
    wds.SimpleShardList(data_dir),
    wds.shuffle(100),
    wds.tarfile_to_samples(1),
    wds.decode(lambda x, y: wds.torch_loads(y)),
    wds.shuffle(batch_size),
    wds.map(wndw_map),
    wds.map(map2),
)
dl = torch.utils.data.DataLoader(
    dpl, num_workers=1,
    batch_size=batch_size,
    collate_fn=collate)
dl_init = torch.utils.data.DataLoader(
    dpl, num_workers=1,
    batch_size=100,
    collate_fn=collate
)
with torch.no_grad():
    data_init = next(iter(dl_init))
    potentials = correlate(data_init['bold']).mean(0)
    del data_init
    del dl_init


model = ConnectomeQueryBiGAN(
    encoder=Encoder(),
    generator=Generator(),
    critic=Critic(),
    qnet=QueryNet()
)
model.set_potentials(potentials.cpu())
model.to(device)

positive_only = UnilateralNormedLoss(nu=positive_only_nu)
community_loss = ModularityLoss(nu=community_nu)


optimizer_EG = torch.optim.Adam(
    params=(
        list(model.encoder.parameters()) +
        list(model.generator.parameters()) +
        list(model.qnet.parameters())
    ),
    lr=lr)
optimizer_C = torch.optim.Adam(
    params=model.critic.parameters(),
    lr=lr)
epochs = Epochs(max_epoch)


noise = torch.randn(8, 2 * embedding_dim + noise_dim)
vidx = 8 *(dim_sequence[0] ** 2) // 10


C_update, EG_update = True, False
C_steps = 0

C_losses, EG_losses = [], []

cur_step = 0
for epoch in epochs:
    dl_iter = iter(dl)
    for data in dl_iter:
        print(f'[ Epoch {epoch} | Step {cur_step} ]')
        x = correlate(data['bold'])
        subject_query = data['sub'].to(device=device, dtype=torch.long)
        task_query = data['task'].to(device=device, dtype=torch.long)
        if cur_step == 0:
            exemplars = {
                'x' : x[:8],
                'subject_query' : subject_query[:8],
                'task_query' : task_query[:8]
            }
        #print(data['__key__'])

        C_loss, EG_loss, (x, z_hat), (x_tilde, z) = model(
            x, subject_query, task_query
        )
        if C_update:
            print('- [ Training critic ]')
            optimizer_C.zero_grad()
            C_loss.backward()
            nonneg, mod = regularisation(model.critic.model.model, x=x)
            nonneg.backward()
            mod.backward()
            C_losses += [C_loss.detach().item()]
            optimizer_C.step()
            C_steps += 1
            if C_steps >= critic_steps_per_generator_step:
                C_update, EG_update = False, True
                C_steps = 0
        elif EG_update:
            print('- [ Training encoder / generator ]')
            optimizer_EG.zero_grad()
            EG_loss.backward()
            EG_losses += [EG_loss.detach().item()]
            nonneg_e, mod = regularisation(model.encoder.model.model, x=x)
            nonneg_g, _ = regularisation(model.generator.model.model)
            nonneg = nonneg_g + nonneg_e
            nonneg.backward()
            mod.backward()
            optimizer_EG.step()
            C_update, EG_update = True, False
        print(f'- [ Wasserstein loss | {EG_loss} ] ')
        print(f'- [ Critic loss with gradient penalty | {C_loss} ] ')
        print(f'- [ Nonnegativity loss | {nonneg} ] ')
        print(f'- [ Modularity loss | {mod} ] ')

        if cur_step % log_interval == 0:
            with torch.no_grad():
                (q_E, q_G), z = model.qnet(
                    subject_query=exemplars['subject_query'],
                    task_query=exemplars['task_query']
                )
                recon = model.reconstruct(
                    exemplars['x'], query_E=q_E, query_G=q_G).cpu()
                gen = model.generate(noise, query=q_G).cpu()

                fig, ax = plt.subplots(3, 8, figsize=(24, 9))
                for i, conn in enumerate(exemplars['x']):
                    ax[0][i].imshow(-conn.cpu(), vmin=-0.4, vmax=0.4, cmap='RdBu')
                    ax[0][i].set_xticks([])
                    ax[0][i].set_yticks([])
                    vlim = recon[i].abs().ravel().sort().values[vidx]
                    ax[1][i].imshow(-recon[i], vmin=-vlim, vmax=vlim, cmap='RdBu')
                    ax[1][i].set_xticks([])
                    ax[1][i].set_yticks([])
                    vlim = gen[i].abs().ravel().sort().values[vidx]
                    ax[2][i].imshow(-gen[i], vmin=-vlim, vmax=vlim, cmap='RdBu')
                    ax[2][i].set_xticks([])
                    ax[2][i].set_yticks([])

                fig.savefig(f'{results}/desc-samples_step-{cur_step:09}.png')
        cur_step += 1
