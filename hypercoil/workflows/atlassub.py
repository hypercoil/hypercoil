"""
The code in this file was contributed from a project by
https://github.com/anna-xu

Archival is a temporary measure while we integrate the underlying principles
and designs into the library.
"""

# Some code adapted from entropy cascade / SWAPR algorithm:
# https://github.com/rciric/hypercoil/blob/diffprog/hypercoil/ ...
# ... workflows/entropycascadeSWAPR.py

# IMPORT
import os
import sys
import torch
import templateflow.api as tflow
from functools import partial
from itertools import chain
from pkg_resources import resource_filename as pkgrf
from indiv_parcel_atlas import AtlasLinearSubjectSpecific
from hypercoil.engine import (
    Epochs,
    MultiplierCascadeSchedule
)
from hypercoil.engine.terminal import ReactiveTerminal
from hypercoil.init.atlas import DirichletInitSurfaceAtlas, CortexSubcortexCIfTIAtlas
from hypercoil.data.functional import random_window, window, identity
from hypercoil.data.transforms import Normalise
from hypercoil.data.wds import torch_wds
from hypercoil.functional import (
    spherical_geodesic, residualise, cmass_coor
)
from hypercoil.functional.domain import MultiLogit
from hypercoil.nn import AtlasLinear
from hypercoil.loss import (
    LossScheme,
    LossApply,
    Entropy,
    Equilibrium,
    Compactness,
    VectorDispersion,
    SecondMoment,
    LogDetCorr,
    HemisphericTether,
    LossArgument,
    UnpackingLossArgument
)
from hypercoil.nn import AtlasLinear
from hypercoil.viz.surf import fsLRAtlasParcels
from torch.optim import Adam

N_VERTICES = 91282
SUBJECTS = (
    100206, 100307, 100408,
    100610, 101006, 101107,
    101309, 101915, 102008,
    102109
)
data_path = '/mnt/andromeda/Data/HCP_SubTaskID_wds/shd-{000000..000179}.tar'
results = '/tmp/indiv_parcels/'
delta_root = f'{results}/deltas'
os.makedirs(results, exist_ok=True)
os.makedirs(delta_root, exist_ok=True)

# For resuming after a crash
resume_epoch = int(sys.argv[2]) # 250 # -1 #
saved_state = sys.argv[1] # '/tmp/indiv_parcels/params-00000250.tar' # None #
deltas_init_scale = 1e-1

print(resume_epoch, saved_state)
#assert 0

# MODEL
def configure_model(n_labels, device='cuda:0', lr=0.05, wd=0, saved_state=None):
    atlas_template = pkgrf(
        'hypercoil',
        'viz/resources/nullexample.nii'
    )
    ref_pointer = '/mnt/pulsar/Data/atlases/atlases/desc-SWAPR_res-000200_atlas.nii'
    atlas_init = CortexSubcortexCIfTIAtlas(
        ref_pointer=ref_pointer,
        mask_L=tflow.get(
            template='fsLR',
            hemi='L',
            desc='nomedialwall',
            density='32k'),
        mask_R=tflow.get(
            template='fsLR',
            hemi='R',
            desc='nomedialwall',
            density='32k'),
        clear_cache=False,
        dtype=torch.float,
        device=device
    )
    # We shouldn't need this
    with torch.no_grad():
        atlas = AtlasLinearSubjectSpecific(
            atlas_init, mask_input=False, kernel_sigma=12,
            domain=MultiLogit(axis=-2, smoothing=.2), max_bin=1000,
            dtype=torch.float, device=device
        )
    atlas.preweight['cortex_L'].requires_grad = False
    atlas.preweight['cortex_R'].requires_grad = False

    plot_atlas = AtlasLinear(atlas_init, mask_input=False,
                        domain=MultiLogit(axis=-2, smoothing=.2),
                        dtype=torch.float, device=device)
    if (saved_state is not None) and (saved_state != 'null'):
        saved_state = torch.load(saved_state, map_location=device)
        atlas.load_state_dict(saved_state['model'])
    else:
        saved_state = None

    lh_coor = atlas.coors[atlas_init.compartments['cortex_L'][atlas.mask]].t()
    rh_coor = atlas.coors[atlas_init.compartments['cortex_R'][atlas.mask]].t()

    lh_mask = atlas_init.compartments['cortex_L'][atlas.mask].unsqueeze(0)
    rh_mask = atlas_init.compartments['cortex_R'][atlas.mask].unsqueeze(0)

    return atlas, plot_atlas, lh_coor, rh_coor, lh_mask, rh_mask, saved_state

# LOSS FUNCTION

# favors parcels being approximately equal size
#equilibrium_nu = 1e8
compactness_nu = 2 # want to increase this over training (Schaeffer paper)
#dispersion_nu = 10 # separate out in beginnign but lower as signals become more separate
# keep symmetry between two hemispheres
#hemisphere_nu = 0.2
#determinant_nu = 0.005
second_moment_nu = 5 # too expensive to put into memory

batch_size = 3
buffer_size = 6
window_size = 500
reactive_slice = 10
lr = 0.005
wd = 0 #2e-6
lr_ephemeral = 0.02
wd_ephemeral = 0 # 1e-7
max_epoch = 6001
data_interval = 5
log_interval = 25
save_interval = 250
n_labels = 200
device = 'cuda:0' # 'cpu' #

epochs = Epochs(max_epoch)
epochs.cur_epoch = resume_epoch
# penalizing this favors a deterministic parcellation (most probability in a couple of voxels)
# Cascading entropy multiplier following the implementation above

##### FAST CASCADE FOR PROTOTYPING #####
#"""
dispersion_nu = 0.5
determinant_nu = .001
compactness_nu = 30
entropy_nu = MultiplierCascadeSchedule(
    epochs=epochs, base=2.0,
    transitions={
        (200, 200): 2.25,
        (500, 500): 2.5,
        (800, 800): 2.75,
        (1100, 1100): 3.0,
        (1300, 1300): 3.5,
        (1500, 1500): 4.0,
        #(900, 900) : 0.25
    }
)
equilibrium_nu = 3e5
#"""

# OPTIMIZER
class AdamEphemeral(Adam):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08,
                 weight_decay=0, amsgrad=False, *, maximize=False): #,
                 #foreach: Optional[bool] = None):
        super().__init__(
            params=params,
            lr=lr,
            betas = betas,
            eps = eps,
            weight_decay=weight_decay,
            amsgrad = amsgrad,
            maximize=maximize,
            #foreach=foreach
        )
        self.ephemeral_index = None
        self.params_ephemeral = {
            'lr' : lr,
            'betas' : betas,
            'eps' : eps,
            'weight_decay' : weight_decay,
            'amsgrad' : amsgrad,
            'maximize' : maximize
        }

    @property
    def ephemeral_state(self):
        state = {}
        if self.ephemeral_index is not None:
            ephemeral = self.param_groups[self.ephemeral_index]['params']
            for p in ephemeral:
                state[p] = self.state[p]
        return state

    def load_ephemeral(self, params, step=None,
                       exp_avg=None, exp_avg_sq=None):
        if isinstance(params, torch.Tensor):
            params = [params]
        params_ephemeral = {'params' : params}
        params_ephemeral.update(self.params_ephemeral)
        self.param_groups += [params_ephemeral]
        self.ephemeral_index = len(self.param_groups) - 1
        if step is not None:
            for i, p in enumerate(params):
                self.state[p] = {}
                self.state[p]['step'] = step[i]
                self.state[p]['exp_avg'] = exp_avg[i]
                self.state[p]['exp_avg_sq'] = exp_avg_sq[i]

    def purge_ephemeral(self):
        if self.ephemeral_index is not None:
            ephemeral = self.param_groups[self.ephemeral_index]['params']
            for p in ephemeral:
                if self.state.get(p) is not None:
                    del self.state[p]
            del self.param_groups[self.ephemeral_index]
            self.ephemeral_index = None

    @torch.no_grad()
    def step(self, closure=None, return_ephemeral_state=True):
        super().step(closure=closure)
        if return_ephemeral_state:
            return self.ephemeral_state


def dl_map(data, window_keys, transpose_keys, window_length):
    data = {
        k: (v.transpose(-2, -1) if k in transpose_keys else v)
        for k, v in data.items()
    }
    window_length, window_start = random_window(data[window_keys[0]], window_length)
    return {
        k: (window(v, window_length, window_start) if k in window_keys else v)
        for k, v in data.items()
    }


# SET UP DATA LOADER
def configure_dataset(data_path, batch_size, buffer_size, window_size):
    wndw = partial(
        dl_map,
        window_keys=('bold', 'confounds', 'tmask'),
        transpose_keys=('bold', 'confounds'),
        window_length=window_size
    )
    maps = {
        'bold' : identity,
        'confounds' : identity,
        'tmask' : identity,
        '__key__' : identity
    }
    ds = torch_wds(
        data_path,
        keys=maps,
        batch_size=batch_size,
        shuffle=buffer_size,
        map=wndw
    )
    # polynomial basis to detrend the data prior to training.
    polybasis = torch.stack([
        torch.arange(500) ** i
        for i in range(3)
    ])
    return ds, polybasis

def transform_sample(sample, mask, atlas, basis, device='cuda:0'):
    """
    Transmute the data into a form suitable for the
    parcellation problem.
    """
    normalise = Normalise()
    X = sample[0][:, :atlas.mask.sum(), :].float()
    X[torch.isnan(X)] = 0
    X = X * mask.view(batch_size, 1, -1)
    gs = X.mean(-2)
    regs = torch.cat((
        basis.tile(X.shape[0], 1, 1),
        gs.unsqueeze(-2)
    ), -2).to(device)
    X = X.to(device)
    regs = regs * mask.view(batch_size, 1, -1).to(device)
    data = residualise(X, regs, driver='gels')
    data = normalise(data)
    return data


def get_keys(key):
    subject = key.split('_')[0].split('-')[-1]
    task = key.split('_')[-1].split('-')[-1]
    return subject, task


def key_mapping(key):
    kmap = []
    umap = set()
    for k in key:
        sub, task = get_keys(k)
        i = [sub, task]
        umap = umap.union({sub, task})
        if task.lower() != 'rest':
            umap = umap.union({'TASK'})
            i += ['TASK']
        kmap.append(tuple(i))
    return kmap, list(umap)


def get_deltas(umap, atlas):
    deltas = {}
    buffers = {}
    for delta in umap:
        path = f'{delta_root}/{delta}.pt'
        buffer_path = f'{delta_root}/{delta}_buffers.pt'
        if os.path.exists(path):
            deltas[delta] = torch.load(path, map_location=device)
            buffers[delta] = torch.load(buffer_path, map_location=device)
        else:
            """
            deltas[delta] = torch.nn.ParameterDict({
                k : torch.nn.Parameter(deltas_init_scale * torch.randn_like(v))
                for k, v in atlas.preweight.items()
            })
            """
            deltas[delta] = torch.nn.ParameterDict({
                k : torch.nn.Parameter(v.clone() + deltas_init_scale * torch.randn_like(v))
                for k, v in atlas.preweight.items()
            })
            bs = [{
                'step': {k: 0},
                'exp_avg' : {k: torch.zeros_like(v)},
                'exp_avg_sq' : {k: torch.zeros_like(v)},
            } for k, v in atlas.preweight.items()]
            buffers[delta] = {'step' : {}, 'exp_avg': {}, 'exp_avg_sq': {}}
            for b in bs:
                for buffer_name in ('step', 'exp_avg', 'exp_avg_sq'):
                    buffers[delta][buffer_name].update(b[buffer_name])
        for v in deltas[delta].values():
            v.requires_grad = True
    return deltas, buffers


def parse_buffers(buffers):
    ephemerals = []
    for buffer in ('step', 'exp_avg', 'exp_avg_sq'):
        e = [list(b[buffer].values()) for b in buffers.values()]
        ephemerals += [list(chain(*e))]
    return tuple(ephemerals)


def load_deltas(key, atlas, opt):
    kmap, umap = key_mapping(key)
    deltas, buffers = get_deltas(umap, atlas)

    ephemerals = [list(d.values()) for d in deltas.values()]
    ephemerals = list(chain(*ephemerals))
    step, exp_avg, exp_avg_sq = parse_buffers(buffers)
    opt.load_ephemeral(
        ephemerals,
        step=step,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq
    )
    return deltas, umap, kmap


def deltas_tensor(deltas, umap, kmap):
    tensor = {
        k: torch.zeros((len(kmap), *v.shape), device=device)
        for k, v in deltas[umap[0]].items()
    }
    for i, maps in enumerate(kmap):
        for m in maps:
            for compartment, data in deltas[m].items():
                tensor[compartment][i] = tensor[compartment][i] + data
    return tensor


def deltas_tensor_compartment(deltas, compartment, umap, kmap):
    ref = deltas[umap[0]][compartment]
    tensor = torch.zeros((len(kmap), *ref.shape), device=device)
    for i, maps in enumerate(kmap):
        for m in maps:
            tensor[i] = tensor[i] + deltas[m][compartment]
    return tensor


def terminal_pretransform(compartment, atlas, deltas, umap, kmap):
    tensor = deltas_tensor_compartment(deltas, compartment, umap, kmap)
    return atlas.domain.image(atlas.preweight[compartment] + tensor)


def get_buffer_from_opt(buffer_name, name, buffers, deltas):
    return {
        compartment: buffers[param][buffer_name]
        for compartment, param in deltas[name].items()
    }


def delta_buffers(umap, buffers, deltas):
    step = {}
    exp_avg = {}
    exp_avg_sq = {}
    buffer_names = (
        (step, 'step'),
        (exp_avg, 'exp_avg'),
        (exp_avg_sq, 'exp_avg_sq')
    )
    for name in umap:
        for d, n in buffer_names:
            d[name] = get_buffer_from_opt(
                buffer_name=n,
                name=name,
                buffers=buffers,
                deltas=deltas
            )
    return step, exp_avg, exp_avg_sq


def save_deltas_and_buffers(umap, deltas, step, exp_avg, exp_avg_sq):
    for name in umap:
        path = f'{delta_root}/{name}.pt'
        buffer_path = f'{delta_root}/{name}_buffers.pt'
        buffers = {
            'step': step[name],
            'exp_avg': exp_avg[name],
            'exp_avg_sq': exp_avg_sq[name]
        }
        torch.save(deltas[name], path)
        torch.save(buffers, buffer_path)


def compartmentalise_data(data, lh_mask, rh_mask):
    """
    Separate out data for the left and right cortical hemispheres.
    """
    shape_L = (data.shape[0], lh_mask.sum(), -1)
    shape_R = (data.shape[0], rh_mask.sum(), -1)
    mask_L = lh_mask.tile(data.shape[0], 1)
    mask_R = rh_mask.tile(data.shape[0], 1)
    data_L = data[mask_L].view(*shape_L)
    data_R = data[mask_R].view(*shape_R)
    return data_L, data_R

# WRITE TRAINING LOOP
ds, polybasis = configure_dataset(
    data_path,
    batch_size=batch_size,
    buffer_size=buffer_size,
    window_size=window_size
)
atlas, plot_atlas, lh_coor, rh_coor, lh_mask, rh_mask, saved_state = configure_model(
    n_labels=n_labels,
    device=device,
    lr=lr,
    wd=0,
    saved_state=saved_state
)
dummy = torch.zeros(0, device=device, requires_grad=True)
opt = AdamEphemeral(
    params=[dummy], lr=lr, weight_decay=wd
)
if saved_state is not None:
    opt.load_state_dict(saved_state['opt'])
#scheduler_lr = torch.optim.lr_scheduler.MultiStepLR(
#    optimizer=opt,
#    milestones=(2750),
#    gamma=0.1,
#    verbose=True)

loss = LossScheme((
    LossScheme((
        Entropy(nu=entropy_nu, name='LeftHemisphereEntropy'),
        Equilibrium(nu=equilibrium_nu, name='LeftHemisphereEquilibrium'),
        Compactness(nu=compactness_nu, radius=100, coor=lh_coor,
                    name='LeftHemisphereCompactness')
    ), apply=lambda arg: arg.lh),
    LossScheme((
        Entropy(nu=entropy_nu, name='RightHemisphereEntropy'),
        Equilibrium(nu=equilibrium_nu, name='RightHemisphereEquilibrium'),
        Compactness(nu=compactness_nu, radius=100, coor=rh_coor,
                    name='RightHemisphereCompactness')
    ), apply=lambda arg: arg.rh),
    LossApply(
        HemisphericTether(nu=hemisphere_nu, name='InterHemisphericTether'),
        apply=lambda arg: UnpackingLossArgument(
            lh=arg.lh,
            rh=arg.rh,
            lh_coor=arg.lh_coor,
            rh_coor=arg.rh_coor)
    ),
    LossApply(
        VectorDispersion(nu=dispersion_nu, metric=spherical_geodesic,
                         name='LeftHemisphereDispersion'),
        apply=lambda arg: cmass_coor(arg.lh, arg.lh_coor, 100).transpose(-2, -1)),
    LossApply(
        VectorDispersion(nu=dispersion_nu, metric=spherical_geodesic,
                         name='RightHemisphereDispersion'),
        apply=lambda arg: cmass_coor(arg.rh, arg.rh_coor, 100).transpose(-2, -1)),
    LossApply(
        LogDetCorr(nu=determinant_nu, psi=0.01, xi=0.01),
        apply=lambda arg: arg.ts)
))
terminal_L = ReactiveTerminal(
    loss=SecondMoment(nu=second_moment_nu,
                      standardise=False,
                      skip_normalise=True,
                      name='LeftHemisphereSecondMoment'),
    slice_target='data',
    slice_axis=-1,
    max_slice=reactive_slice
)

terminal_R = ReactiveTerminal(
    loss=SecondMoment(nu=second_moment_nu,
                      standardise=False,
                      skip_normalise=True,
                      name='RightHemisphereSecondMoment'),
    slice_target='data',
    slice_axis=-1,
    max_slice=reactive_slice
)

network_cmap = pkgrf(
    'hypercoil',
    'viz/resources/cmap_network.nii'
)
views = ('medial', 'lateral')

no_data = True
base = float('nan') * torch.zeros((batch_size, N_VERTICES, window_size))
basemask = torch.zeros((batch_size, window_size), dtype=torch.bool)
for epoch in epochs:
    #print(epoch)
    #print(atlas.weight['cortex_L'].max(-2))
    if epoch % data_interval == 0 or no_data:
        for sample in ds:
            sample[-2] = ~torch.isnan(sample[-2]) * sample[-2].bool()
            mask = basemask.clone()
            mask[..., :sample[-2].size(-1)] = sample[-2]
            data = base.clone()
            data[..., :sample[0].size(-1)] = sample[0]
            sample[0] = data
            data = transform_sample(sample, mask, atlas, polybasis, device)
            key = sample[-1]
            mask = mask.to(device)
            break
        no_data = False

    data_L, data_R = compartmentalise_data(data, lh_mask, rh_mask)
    # TODO: Load ephemeral params here.
    deltas, umap, kmap = load_deltas(key, atlas, opt)
    opt.param_groups[opt.ephemeral_index]['weight_decay'] = wd_ephemeral
    opt.param_groups[opt.ephemeral_index]['lr'] = lr_ephemeral
    dtensor = deltas_tensor(deltas, umap, kmap)
    parcel_data = atlas(data, delta=dtensor)

    terminal_L.pretransforms = {
        'weight' : partial(
            terminal_pretransform,
            atlas=atlas, deltas=deltas, umap=umap, kmap=kmap
        )
    }
    terminal_R.pretransforms = {
        'weight' : partial(
            terminal_pretransform,
            atlas=atlas, deltas=deltas, umap=umap, kmap=kmap
        )
    }

    #print(torch.softmax(atlas.preweight['cortex_L'] + dtensor['cortex_L'], -2))
    #print(torch.softmax(atlas.preweight['cortex_L'] + dtensor['cortex_L'], -2).max(-2))
    arg = LossArgument(
        ts=parcel_data,
        lh=torch.softmax(atlas.preweight['cortex_L'] + dtensor['cortex_L'], -2),
        rh=torch.softmax(atlas.preweight['cortex_R'] + dtensor['cortex_R'], -2),
        lh_coor=lh_coor, rh_coor=rh_coor
    )
    arg_L = {'data': data_L, 'weight': 'cortex_L'}
    arg_R = {'data': data_R, 'weight': 'cortex_R'}

    out = terminal_L(arg=arg_L) #, axis_mask=mask)
    print(f'- {terminal_L.loss} : {out}')
    out = terminal_R(arg=arg_R) #, axis_mask=mask)
    print(f'- {terminal_R.loss} : {out}')
    out = loss(arg, verbose=True)
    print(f'[ Epoch {epoch} | Total loss {out} ]\n')
    out.backward()
    #TODO: Capture, purge, and save ephemeral parameters below.
    ephemeral_buffers = opt.step()
    opt.zero_grad()
    step, exp_avg, exp_avg_sq = delta_buffers(umap, ephemeral_buffers, deltas)
    save_deltas_and_buffers(umap, deltas, step, exp_avg, exp_avg_sq)
    opt.purge_ephemeral()
    del dtensor, deltas, ephemeral_buffers
    torch.cuda.empty_cache()
    #scheduler_lr.step()

    if epoch % log_interval == 0:
        rest = torch.load(f'{delta_root}/REST.pt')
        for sub in SUBJECTS:
            with torch.no_grad():
                try:
                    delta = torch.load(f'{delta_root}/{sub}.pt')
                except FileNotFoundError:
                    continue
                for c in ('cortex_L', 'cortex_R'):
                    plot_atlas.preweight[c][:] = (
                        atlas.preweight[c] + rest[c] + delta[c])
                plotter = fsLRAtlasParcels(plot_atlas)
                #print(plot_atlas.atlas.compartments['cortex_L'].device, plot_atlas.atlas.decoder['cortex_L'].device)
                plotter(
                    cmap=network_cmap,
                    views=views,
                    save=f'{results}/sub-{sub}_epoch-{epoch:08}'
                )
        del rest, delta

    if epoch % save_interval == 0:
        torch.save({
            'model': atlas.state_dict(),
            'opt': opt.state_dict(),
        }, f'{results}/params-{epoch:08}.tar')
        assert 0
