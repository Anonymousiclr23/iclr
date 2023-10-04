import sys, os

import  argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--epoch', type=int, help='epoch number', default=100)
argparser.add_argument('--imgc', type=int, help='imgc', default=1)
argparser.add_argument('--outc', type=int, help='outc', default=2)
argparser.add_argument('--batch_size', type=int, help='batch size', default=64)
argparser.add_argument('--start_lr', type=float, help='initial learning rate', default=1e-3)
argparser.add_argument('--end_lr', type=float, help='fianl learning rate', default=1e-6)
argparser.add_argument("--data_folder", type=str, help='folder for the data', default="/scratch/groups/jonfan/UNet/data/data_generation_52_thick_8bar_Si/30k_new_wmin625")
argparser.add_argument("--total_sample_number", type=int, help="total number of training and testing samples to take from the npy file (in case you don't want to use all the data there)", default=None)
argparser.add_argument("--arch", type=str, help='architecture of the learner', default="Fourier")
argparser.add_argument("--HIDDEN_DIM", type=int, help='width of Unet, i.e. number of kernels of first block', default=64)
argparser.add_argument("--model_saving_path", type=str, help="the root dir to save checkpoints", default="") 
argparser.add_argument("--model_name", type=str, help="name for the model, used for storing under the model_saving_path", default="test")
argparser.add_argument("--exp_decay", type=float, help="exponential decay of learning rate, update per epoch", default=0.98)
argparser.add_argument("--lr_update_steps", type= int, help = "how many gradient backprops to update the learning rate", default=5000)
argparser.add_argument("--continue_train", type=str, help = "if true, continue train from continue_epoch", default='False')
argparser.add_argument("--ALPHA", type=float, help="negative slope of leaky relu", default=0.05)
argparser.add_argument("--weight_decay", type=float, help="l2 regularization coeff", default=1e-4)
argparser.add_argument("--ratio", type=float, help="relative weight of physical regularizer", default = 0.5)
argparser.add_argument("--phys_start_epoch", type=int, help="starting epoch of physical regularizer", default = 1)
argparser.add_argument("--kernel_size", type=int, help="conv layer kernel size", default=3)
argparser.add_argument("--f_modes", type=int, help="number of lowest fourier terms to keep and transform", default=20)
argparser.add_argument("--num_fourier_layers", type=int, help="number of lowest fourier terms to keep and transform", default=10)
argparser.add_argument("--domain_sizex", type=int, help="number of pixels in x direction of subdomain", default=32)
argparser.add_argument("--domain_sizey", type=int, help="number of pixels in y direction of subdomain", default=32)
argparser.add_argument("--f_padding", type=int, help="padding for non-periodic b.c.", default = 0)
argparser.add_argument("--data_mult", type=float, help="multiplier for the data", default = 1)
argparser.add_argument("--source_mult", type=float, help="multiplier for source", default = 1)

#args relating to distributed training and memory 
argparser.add_argument("--num_GPUs", type=int, help = "how many GPUs are visible", default=1)
argparser.add_argument("--visible_GPUs", type=str, help = "which GPUs are visible and therefore usable", default="0")

# args for physics training:
argparser.add_argument("--inner_weight", type=float, help="weight for inner physics loss term", default = 1)
argparser.add_argument("--bc_weight", type=float, help="weight for bounadry physics loss term", default = 1)
argparser.add_argument("--data_weight", type=float, help="weight for data loss term", default = 1)

args = argparser.parse_args()

assert args.batch_size % args.num_GPUs == 0

print("verify visible: ", args.visible_GPUs)
os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_GPUs

# first set which GPU devices are visible then import jax
import jax

sys.path.append("../util")
from JAX_SM_FNO_source_conv import FNO_multimodal_2d
from simulation_dataset_JAX_source_only import SimulationDataset

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import  torch

import pandas as pd

from jax import numpy as jnp
from jax import random
from jax.example_libraries import optimizers
from jax.tree_util import tree_map

import optax 
import equinox as eqx
from functools import partial

import matplotlib 
matplotlib.use('agg')
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import timeit

import cloudpickle
import pickle

############### physical constants and helper functions for calculating physics residue ##################

C_0 = 299792458.13099605
EPSILON_0 = 8.85418782e-12
MU_0 = 1.25663706e-6
dL = 6.25e-9
wavelength = 1050e-9
omega = 2 * jnp.pi * C_0 / wavelength
n_air=1.
n_Si=3.567390909090909
n_sub=1.45

@eqx.filter_jit
def Hz_to_Ex(Hz_R, Hz_I, dL, omega, EPSILON_0 = EPSILON_0):
    Ex_R = -(Hz_I[:, 1:, 1:-1] - Hz_I[:, 0:-1, 1:-1])/dL/omega/EPSILON_0 # The returned Ex is corresponding to Ex_ceviche[:, 0:-1]
    Ex_I = (Hz_R[:, 1:, 1:-1] - Hz_R[:, 0:-1, 1:-1])/dL/omega/EPSILON_0

    return jnp.stack((Ex_R, Ex_I), axis = 1)

@eqx.filter_jit
def Hz_to_Ey(Hz_R, Hz_I, dL, omega, EPSILON_0 = EPSILON_0):
    Ey_R = (Hz_I[:,1:-1,1:] - Hz_I[:,1:-1,:-1])/dL/omega/EPSILON_0 # The returned Ey is corresponding to Ey_ceviche[0:-1, :]
    Ey_I = -(Hz_R[:,1:-1,1:] - Hz_R[:,1:-1,:-1])/dL/omega/EPSILON_0
    return jnp.stack((Ey_R, Ey_I), axis = 1)

@eqx.filter_jit
def E_to_Hz(Ey_R, Ey_I, Ex_R, Ex_I, dL, omega, MU_0 = MU_0):
    Hz_R = ((Ey_I[:, :, 1:] - Ey_I[:, :, 0:-1]) - (Ex_I[:, 1:, :] - Ex_I[:, 0:-1, :]))/dL/omega/MU_0
    Hz_I = -((Ey_R[:, :, 1:] - Ey_R[:, :, 0:-1]) - (Ex_R[:, 1:, :] - Ex_R[:, 0:-1, :]))/dL/omega/MU_0
    return jnp.stack((Hz_R, Hz_I), axis = 1) # -Hy[1:, :]

@eqx.filter_jit
def H_to_H_src(Hz_R, Hz_I, dL, omega, source, EPSILON_0 = EPSILON_0, MU_0 = MU_0):
    FD_Ex = Hz_to_Ex(Hz_R, Hz_I, dL, omega, EPSILON_0)
    FD_Ey = Hz_to_Ey(Hz_R, Hz_I, dL, omega, EPSILON_0)
    FD_H = E_to_Hz(FD_Ey[:, 0], FD_Ey[:, 1], FD_Ex[:, 0], FD_Ex[:, 1], dL, omega, MU_0)

    # additional term: 1j*wavelength/(2*jnp.pi)*C_0*EPSILON_0*source[:,:,1:-1,1:-1]
    source_vector = jnp.stack(((wavelength/(2*jnp.pi*dL))**2*source[:,1:-1,1:-1, 0], 
                               (wavelength/(2*jnp.pi*dL))**2*source[:,1:-1,1:-1, 1]), axis = 1)
    return FD_H + source_vector

@eqx.filter_jit
def H_to_bc_src(Hz_R, Hz_I, dL, omega, EPSILON_0 = EPSILON_0):
    Hz = Hz_R + 1j*Hz_I
    top_bc = (Hz[:,0,1:-1]-Hz[:,1,1:-1])+1j*2*jnp.pi/wavelength*dL*1/2*(Hz[:,0,1:-1]+Hz[:,1,1:-1])
    bottom_bc = (Hz[:,-1,1:-1]-Hz[:,-2,1:-1])+1j*2*jnp.pi/wavelength*dL*1/2*(Hz[:,-1,1:-1]+Hz[:,-2,1:-1])
    left_bc = (Hz[:,1:-1,0]-Hz[:,1:-1,1])+1j*2*jnp.pi/wavelength*dL*1/2*(Hz[:,1:-1,0]+Hz[:,1:-1,1])
    right_bc = (Hz[:,1:-1,-1]-Hz[:,1:-1,-2])+1j*2*jnp.pi/wavelength*dL*1/2*(Hz[:,1:-1,-1]+Hz[:,1:-1,-2])
    return jnp.stack((jnp.real(top_bc), jnp.imag(top_bc), jnp.real(bottom_bc), jnp.imag(bottom_bc),\
                        jnp.real(left_bc), jnp.imag(left_bc), jnp.real(right_bc), jnp.imag(right_bc)), axis = 1)

def plot_helper(data,step,path):
    plt.figure()
    ax = plt.gca()
    im = ax.imshow(data)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.title(f"epoch{step}")
    plt.savefig(path, transparent=True)
    plt.close()

def regConstScheduler(epoch, args, last_epoch_data_loss, last_epoch_physical_loss):
    if(epoch<args.phys_start_epoch):
        return 0
    else:
        return args.ratio*last_epoch_data_loss/last_epoch_physical_loss

###################################################################################################

def main(args,seed):
    key = jax.random.PRNGKey(seed)

    # Loading and splitting dataset
    model_path = args.model_saving_path + args.model_name + "/" + \
                                          "_domain_size_" + str(args.domain_sizex) + "_"+ str(args.domain_sizey) + \
                                          "_fmodes_" + str(args.f_modes) + \
                                          "_flayers_" + str(args.num_fourier_layers) + \
                                          "_Hidden_" + str(args.HIDDEN_DIM) + \
                                          "_f_padding_" + str(args.f_padding) + \
                                          "_batch_size_" + str(args.batch_size) + "_lr_" + str(args.start_lr)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
        os.mkdir(model_path+'/plots')

    ds = SimulationDataset(args.data_folder, total_sample_number = args.total_sample_number, data_mult=args.data_mult)
    train_ds, test_ds = random_split(ds, [int(0.9*len(ds)), len(ds) - int(0.9*len(ds))])

    start_epoch = 0
    if (args.continue_train == "True"):
        df = pd.read_csv(model_path + '/'+'df.csv')
        print("Restoring weights from ", model_path+"/last_model.pt", flush=True)
        _file = open(model_path+"/last_model.pt", 'rb')
        checkpoint = pickle.load(_file)
        start_epoch=checkpoint['epoch']
        opt_state = checkpoint['opt_state'],
        optim = checkpoint['optim']
        print(f"start_epoch is {start_epoch}")
        model = eqx.tree_deserialise_leaves("last_model.eqx", model_original)
    else:
        df = pd.DataFrame(columns=['epoch','train_loss', 'train_phys_reg', 'test_loss', 'test_phys_reg'])
        model = FNO_multimodal_2d(args, key = key)
        
        # @optax.inject_hyperparams
        def optimizer(start_lr, end_lr ,steps):
            scheduler = optax.exponential_decay(
                init_value=start_lr, 
                transition_steps=steps,
                decay_rate=0.99,
                end_value=end_lr)

            return optax.chain(
                # optax.clip_by_global_norm(0.3),  # Clip by the gradient by the global norm.
                optax.scale_by_adam(),  # Use the updates from adam.
                optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
                # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
                optax.scale(-1.0)
            )
	   # calculate the transition steps based on 0.99 decay rate
        total_steps = args.epoch*(0.9*len(train_ds)/args.batch_size)
        update_steps = int(jnp.log(args.end_lr/args.start_lr)/jnp.log(0.99))
        transition_steps = int(total_steps/update_steps)
        optim = optimizer(args.start_lr, args.end_lr, transition_steps)
        
        opt_state = optim.init(eqx.filter(model, eqx.is_array))
        print(f"start_lr: {args.start_lr}, end_lr: {args.end_lr}, transition steps: {transition_steps}, decay: 0.99")

    tmp = jax.tree_util.tree_flatten(eqx.filter(model, eqx.is_array))
    def get_array_size(x):
        if 'shape' in x.__dir__():
            return jnp.prod(jnp.asarray(x.shape))
        else:
            _sum = 0
            if '__len__' in x.__dir__() and (len(x) > 0):
                for a in x:
                    _sum += get_array_size(a)
            return _sum
    num = sum(map(lambda x: get_array_size(x), tmp))
    # print(model)
    print('Total trainable tensors:', num, flush=True)
    with open(model_path + '/'+'config.txt', 'w') as f:
        f.write('\n'.join(sys.argv[1:]))
        f.write(model.__str__())
        f.write(f'Total trainable tensors: {num}')

    train_loader = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True)
    total_step = len(train_loader)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True)

    # some helper function for data mapping reduce:
    replicate_array = lambda x: jnp.broadcast_to(x, (args.num_GPUs,) + x.shape) if eqx.is_array(x) else x
    extract_array = lambda x: x[0] if eqx.is_array(x) else x
    map_psum = lambda x: jax.lax.psum(x, 'batch') if eqx.is_array(x) else x

    def compute_loss_only(model, source, top_bc, bottom_bc, left_bc, right_bc, y, source_mult):
        pred_y, bc, grid = eqx.filter_vmap(model.eval_forward)(source_mult*source, top_bc, bottom_bc, left_bc, right_bc)

        data_loss = MAE_loss(pred_y, y)
        FD_Hy = H_to_H_src(pred_y[:, :, :, 0], pred_y[:, :, :, 1], dL, omega, source)
        # print("sh: ", FD_Hy.shape, logits.shape)
        phys_regR = MAE_loss(FD_Hy[:, 0], pred_y[:, 1:-1, 1:-1, 0])
        phys_regI = MAE_loss(FD_Hy[:, 1], pred_y[:, 1:-1, 1:-1, 1])
        reg_inner = 0.5*(phys_regR + phys_regI)

        return data_loss, reg_inner, pred_y, bc, grid

    @eqx.filter_jit
    def MAE_loss(x1, x2):
        return jnp.mean(jnp.abs(x1-x2))/jnp.mean(jnp.abs(x2))

    @eqx.filter_value_and_grad(has_aux=True)
    def compute_loss_and_grad(model, source, top_bc, bottom_bc, left_bc, right_bc, y, dL, omega, reg_norm, data_weight, inner_weight, bc_weight, source_mult):
        pred_y = eqx.filter_vmap(model)(source_mult*source, top_bc, bottom_bc, left_bc, right_bc)

        data_loss = data_weight*MAE_loss(pred_y, y)

        FD_Hy = H_to_H_src(pred_y[:, :, :, 0], pred_y[:, :, :, 1], dL, omega, source)
        # print("sh: ", FD_Hy.shape, logits.shape)
        phys_regR = MAE_loss(FD_Hy[:, 0], pred_y[:, 1:-1, 1:-1, 0])
        phys_regI = MAE_loss(FD_Hy[:, 1], pred_y[:, 1:-1, 1:-1, 1])
        reg_inner = inner_weight*0.5*(phys_regR + phys_regI) 

        # FD_Hy_gt = H_to_H_src(y[:, :, :, 0], y[:, :, :, 1], dL, omega, source)
        # reg_inner_gt = 1/2 * MAE_loss(FD_Hy_gt[:, 0], y[:, 1:-1, 1:-1, 0]) + 1/2 * MAE_loss(FD_Hy_gt[:, 1], y[:, 1:-1, 1:-1, 1])

        # physical loss for the boundary:
        bc_gt = H_to_bc_src(y[:,:,:,0], y[:,:,:,1],dL, omega)
        bc_pred = H_to_bc_src(pred_y[:,:,:,0], pred_y[:,:,:,1], dL,omega)
        phys_reg_bc = bc_weight*MAE_loss(bc_pred, bc_gt)

        physics_loss = reg_norm*(reg_inner+phys_reg_bc)

        aux = (data_loss, reg_inner, phys_reg_bc)

        return data_loss+physics_loss, aux

    @eqx.filter_pmap(axis_name='batch')
    def train_step(rep_models, source, top_bc, bottom_bc, left_bc, right_bc, y, replicated_opt_state, dL, omega, reg_norm, data_weight, inner_weight, bc_weight, source_mult):
        (loss, aux), grads = compute_loss_and_grad(rep_models, source, top_bc, bottom_bc, left_bc, right_bc, y, dL, omega, reg_norm, data_weight, inner_weight, bc_weight, source_mult)
        grads = tree_map(map_psum, grads)
        
        updates, replicated_opt_state = optim.update(grads, replicated_opt_state)
        rep_models = eqx.apply_updates(rep_models, updates)

        return loss, rep_models, replicated_opt_state, aux

    @eqx.filter_pmap(axis_name='batch')
    def eval_step(rep_models, source, top_bc, bottom_bc, left_bc, right_bc, y, source_mult):
        loss, reg_inner, pred_y, bc, grid = compute_loss_only(rep_models, source, top_bc, bottom_bc, left_bc, right_bc, y, source_mult)
        
        return loss, reg_inner, pred_y, bc, grid

    # we need to replicate our model so that each pytree leaf leads with dimension equal to number of GPUs:
    replicated_models = tree_map(replicate_array, model)
    replicated_opt_state = tree_map(replicate_array, opt_state)

    # train loop:
    start_epoch=0
    best_loss = 1e4
    last_epoch_data_loss = 1
    last_epoch_physical_loss = 1 

    running_average_data_loss = 1
    running_average_inner_residue = 10 # (it's hard to set a initial value for physics residue but it doesn't matter)
    for step in range(start_epoch, args.epoch):
        print("epoch: ", step)
        epoch_start_time = timeit.default_timer()
        reg_norm = regConstScheduler(step, args, last_epoch_data_loss, last_epoch_physical_loss)
        for idx, sample_batched in enumerate(train_loader):
            y_batch_train, source_batch_train, top_bc_train, bottom_bc_train, left_bc_train, right_bc_train = jnp.asarray(sample_batched['field']), jnp.asarray(sample_batched['source']), jnp.asarray(sample_batched['top_bc']), jnp.asarray(sample_batched['bottom_bc']), jnp.asarray(sample_batched['left_bc']), jnp.asarray(sample_batched['right_bc'])
            
            # reshape data so that first dim is equal to number of devices
            y_batch_train = y_batch_train.reshape(args.num_GPUs,int(args.batch_size/args.num_GPUs),y_batch_train.shape[1],y_batch_train.shape[2],y_batch_train.shape[3])
            source_batch_train = source_batch_train.reshape(args.num_GPUs,int(args.batch_size/args.num_GPUs),source_batch_train.shape[1],source_batch_train.shape[2],source_batch_train.shape[3])
            top_bc_train = top_bc_train.reshape(args.num_GPUs,int(args.batch_size/args.num_GPUs),top_bc_train.shape[1],top_bc_train.shape[2],top_bc_train.shape[3])
            bottom_bc_train = bottom_bc_train.reshape(args.num_GPUs,int(args.batch_size/args.num_GPUs),bottom_bc_train.shape[1],bottom_bc_train.shape[2],bottom_bc_train.shape[3])
            left_bc_train = left_bc_train.reshape(args.num_GPUs,int(args.batch_size/args.num_GPUs),left_bc_train.shape[1],left_bc_train.shape[2],left_bc_train.shape[3])
            right_bc_train = right_bc_train.reshape(args.num_GPUs,int(args.batch_size/args.num_GPUs),right_bc_train.shape[1],right_bc_train.shape[2],right_bc_train.shape[3])

            loss, replicated_models, replicated_opt_state, aux = train_step(replicated_models, source_batch_train, top_bc_train, bottom_bc_train, left_bc_train, right_bc_train, y_batch_train, replicated_opt_state, dL, omega, reg_norm, args.data_weight, args.inner_weight, args.bc_weight, args.source_mult)

            running_average_data_loss = 0.99*running_average_data_loss+0.01*jnp.mean(loss).item()
            running_average_inner_residue = 0.99*running_average_inner_residue+0.01*jnp.mean(aux[1]).item()
            
            if idx%100 ==0:
                print('Epoch [{}/{}], Step [{}/{}], data Loss: {:.4f}'.format(
                    step + 1,
                    args.epoch,
                    idx + 1,
                    total_step,
                    jnp.mean(loss).item()
                    )
                )

        model = tree_map(extract_array, replicated_models)
        opt_state = tree_map(extract_array, replicated_opt_state)
        
        # save checkpoint
        eqx.tree_serialise_leaves(model_path+f"/last_model.eqx", model)
        checkpoint = {
                        'epoch': step,
                        'opt_state': opt_state,
                        'optimizer': optim
                    }
        with open(model_path+"/last_model.pt", 'wb') as handle:
            checkpoint = cloudpickle.dumps(checkpoint)
            pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # to save time, approx train loss here:
        train_loss = running_average_data_loss
        train_phys_reg = running_average_inner_residue

        # test eval
        test_loss = 0
        test_phys_reg = 0
        for idx, sample_batched in enumerate(test_loader):
            y_batch_test, source_batch_test, top_bc_test, bottom_bc_test, left_bc_test, right_bc_test = jnp.asarray(sample_batched['field']), jnp.asarray(sample_batched['source']), jnp.asarray(sample_batched['top_bc']), jnp.asarray(sample_batched['bottom_bc']), jnp.asarray(sample_batched['left_bc']), jnp.asarray(sample_batched['right_bc'])
            y_batch_test = y_batch_test.reshape(args.num_GPUs,int(args.batch_size/args.num_GPUs),y_batch_test.shape[1],y_batch_test.shape[2],y_batch_test.shape[3])
            source_batch_test = source_batch_test.reshape(args.num_GPUs,int(args.batch_size/args.num_GPUs),source_batch_test.shape[1],source_batch_test.shape[2],source_batch_test.shape[3])
            top_bc_test = top_bc_test.reshape(args.num_GPUs,int(args.batch_size/args.num_GPUs),top_bc_test.shape[1],top_bc_test.shape[2],top_bc_test.shape[3])
            bottom_bc_test = bottom_bc_test.reshape(args.num_GPUs,int(args.batch_size/args.num_GPUs),bottom_bc_test.shape[1],bottom_bc_test.shape[2],bottom_bc_test.shape[3])
            left_bc_test = left_bc_test.reshape(args.num_GPUs,int(args.batch_size/args.num_GPUs),left_bc_test.shape[1],left_bc_test.shape[2],left_bc_test.shape[3])
            right_bc_test = right_bc_test.reshape(args.num_GPUs,int(args.batch_size/args.num_GPUs),right_bc_test.shape[1],right_bc_test.shape[2],right_bc_test.shape[3])

            if idx == 0:
                loss, reg_inner, pred_y, bc, grid = eval_step(replicated_models, source_batch_test, top_bc_test, bottom_bc_test, left_bc_test, right_bc_test, y_batch_test, args.source_mult)
                # pred_y = pred_y*mean_bc
                plot_helper(pred_y[0,0,:,:,0], step, model_path+"/plots/epoch_"+str(step)+"_output.png")
                plot_helper(y_batch_test[0,0,:,:,0]-pred_y[0,0,:,:,0], step, model_path+"/plots/epoch_"+str(step)+"_error.png")
                plot_helper(y_batch_test[0,0,:,:,0], step, model_path+"/plots/epoch_"+str(step)+"_gt.png")
                plot_helper(source_batch_test[0,0,:,:,0], step, model_path+"/plots/epoch_"+str(step)+"_source.png")
            else:
                loss, reg_inner, pred_y, bc, grid = eval_step(replicated_models, source_batch_test, top_bc_test, bottom_bc_test, left_bc_test, right_bc_test, y_batch_test, args.source_mult)

            test_loss += jnp.mean(loss).item()
            test_phys_reg += jnp.mean(reg_inner).item()

        test_loss /= len(test_loader)
        test_phys_reg /= len(test_loader)
        last_epoch_data_loss = test_loss
        last_epoch_physical_loss = test_phys_reg

        print('train loss: %.5f, test loss: %.5f' % (train_loss, test_loss), flush=True)
        
        new_df = pd.DataFrame([[step+1,0, train_loss, train_phys_reg, test_loss, test_phys_reg]], \
                            columns=['epoch', 'lr', 'train_loss', 'train_phys_reg', 'test_loss', 'test_phys_reg'])
        df = pd.concat([df,new_df])

        df.to_csv(model_path + '/'+'df.csv',index=False)

        if(test_loss<best_loss):
            best_loss = test_loss
            eqx.tree_serialise_leaves(model_path+f"/best_model.eqx", model)
            checkpoint = {
                            'epoch': step,
                            'opt_state': opt_state,
                            'optimizer': optim
                        }
            with open(model_path+"/best_model.pt", 'wb') as handle:
                checkpoint = cloudpickle.dumps(checkpoint)
                pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)

        epoch_stop_time = timeit.default_timer()
        print("epoch run time:", epoch_stop_time-epoch_start_time)

if __name__ == '__main__':
    print("jax.devices: ", jax.devices())
    main(args,42)
