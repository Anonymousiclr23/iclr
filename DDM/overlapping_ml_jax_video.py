import os, sys
import argparse
from tqdm import tqdm
from functools import partial
from argparse import Namespace
argparser = argparse.ArgumentParser()

# parameters of each sub-domain model:
argparser.add_argument('--imgc', type=int, help='imgc', default=1)
argparser.add_argument('--outc', type=int, help='outc', default=2)
argparser.add_argument('--batch_size', type=int, help='batch size', default=64)
argparser.add_argument('--lr', type=float, help='initial learning rate', default=1e-3)
argparser.add_argument("--data_folder", type=str, help='folder for the data', default="")
argparser.add_argument("--total_shapex", type=int, help='total size in x', default=0)
argparser.add_argument("--total_shapey", type=int, help='total size in y', default=0)
argparser.add_argument("--periodic", type=int, help='if == 1, device is half periodic', default=0)

argparser.add_argument("--data_mult", type=float, help='multiplier for the data', default=1)
argparser.add_argument("--source_mult", type=float, help='additional multiplier for the source', default=10)
argparser.add_argument("--total_sample_number", type=int, help="total number of training and testing samples to take from the npy file (in case you don't want to use all the data there)", default=None)
argparser.add_argument("--arch", type=str, help='architecture of the learner', default="Fourier")
# argparser.add_argument('--NUM_DOWN_CONV', type=int, help='number of down conv blocks in Unet', default=6)
argparser.add_argument("--HIDDEN_DIM", type=int, help='width of Unet, i.e. number of kernels of first block', default=64)
argparser.add_argument("--model_saving_path", type=str, help="the root dir to save checkpoints", default="") 
# argparser.add_argument("--model_name", type=str, help="name for the model, used for storing under the model_saving_path", default="test")
argparser.add_argument("--mat_model_name", type=str, help="model for high contrast grascale material", default="")
argparser.add_argument("--src_model_name", type=str, help="model for source", default="")
argparser.add_argument("--pml_model_name", type=str, help="model for pml", default="")      

argparser.add_argument("--kernel_size", type=int, help="conv layer kernel size", default=3)
argparser.add_argument("--f_modes", type=int, help="number of lowest fourier terms to keep and transform", default=20)
argparser.add_argument("--num_fourier_layers", type=int, help="number of lowest fourier terms to keep and transform", default=10)
argparser.add_argument("--domain_sizex", type=int, help="number of pixels in x direction of subdomain", default=16)
argparser.add_argument("--domain_sizey", type=int, help="number of pixels in y direction of subdomain", default=16)
argparser.add_argument("--f_padding", type=int, help="padding for non-periodic b.c.", default = 0)
argparser.add_argument("--visible_GPUs", type=str, help = "which GPUs are visible and therefore usable", default="0")

#parameters of overlapping DDM: 
argparser.add_argument("--overlap_pixels", type=int, help="the # of overlapping pixels of adjacent subdomain", default = 10)
argparser.add_argument("--starting_x", type=int, help="index of starting x", default = 0)
argparser.add_argument("--starting_y", type=int, help="index of starting y", default = 0)
argparser.add_argument("--DDM_iters", type=int, help="number of iterations for the overlapping schwarz's algorithm", default = 10)
argparser.add_argument("--bc_func", type=str, help="what boundary update function to use. refer to utils/DDM_util.py", default = "")
argparser.add_argument("--float_bits", type=int, help="whether use 32 bit or 64 bit floating number for computation", default = 64)
argparser.add_argument("--momentum", type=float, help="next_iter_bc = momentum*last_iter_bc + (1-momentum)*new_bc", default = 0)
argparser.add_argument("--pml_thickness", type=int, help="total number of pixels of pml in each direction", default = 40)

# args for simulation setup (constants)
argparser.add_argument("--dL", type=float, help="simulation pixel size, in [m]", default = 6.25e-9)
argparser.add_argument("--wl", type=float, help="simulation wavelength, in [m]", default = 1050e-9)

# args for plotting:
argparser.add_argument("--div_k", type=int, help="plot per div_k iterations", default = 5)
argparser.add_argument("--write_video", type=int, help="if == 1, write video", default = 0)
argparser.add_argument("--seed", type=int, help="seed", default = 0)

args = argparser.parse_args() 
print("verify visible: ", args.visible_GPUs)
os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_GPUs

import jax
import equinox as eqx
import jax.numpy as jnp
import torch

from    torch import optim
import  numpy as np
import pandas as pd

from torch.utils.data import random_split, DataLoader 
import time

from JAX_DDM_util import *

from JAX_SM_FNO_mat_conv import FNO_multimodal_2d as SM_FNO_mat
from JAX_SM_FNO_source_conv import FNO_multimodal_2d as SM_FNO_src
from JAX_SM_FNO_pml import FNO_multimodal_2d as SM_FNO_pml

from DDM_dataset import DDM_Dataset
from matplotlib import pyplot as plt
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cv2

EPSILON_0 = 8.85418782e-12        # vacuum permittivity
MU_0 = 1.25663706e-6              # vacuum permeability
C_0 = 1 / np.sqrt(EPSILON_0 * MU_0)  # speed of light in vacuum
ETA_0 = np.sqrt(MU_0 / EPSILON_0)    # vacuum impedance
Q_e = 1.602176634e-19             # funamental charge

def get_non_zero_indices(batched_data):
    reduce_dims = tuple(range(1,len(batched_data.shape)))
    data = jnp.sum(batched_data, axis=reduce_dims)
    return tuple(jnp.nonzero(data)[0].tolist())

@partial(jax.jit, static_argnums = [0,1,2,3,4])
def combine_batch(total_size, mat_indices, src_indices, pml_indices, shape, mat_batch, src_batch, pml_batch):
    combined = []
    if len(mat_indices)==0:
        mat_indices = [-1]
    if len(src_indices)==0:
        src_indices = [-1]
    if len(pml_indices)==0:
        pml_indices = [-1]

    count = 0
    head_mat=0
    head_src=0
    head_pml=0
    while count < total_size:
        if count==mat_indices[head_mat]:
            combined.append(mat_batch[head_mat])
            head_mat = min(len(mat_indices)-1, head_mat+1)
            count += 1
        elif count==src_indices[head_src]:
            combined.append(src_batch[head_src])
            head_src = min(len(src_indices)-1, head_src+1)
            count += 1
        elif count==pml_indices[head_pml]:
            combined.append(pml_batch[head_pml])
            head_pml = min(len(pml_indices)-1, head_pml+1)
            count += 1
        else:
            raise ValueError("no idx in all three lists for ", count)

    logits_RI = jnp.stack(combined).reshape(shape)
    return logits_RI[:,:,:,0] + 1j*logits_RI[:,:,:,1]

def reconstruct_args(f):
    args = {'outc': 2}
    key = None
    while True:
        line = f.readline()
        if line[:2] == "--":
            key = line[2:-1]
        elif "FNO_multimodal_2d" in line:
            args[key] = line.split('F')[0]
            break
        elif key in ['f_modes', 'HIDDEN_DIM', 'num_fourier_layers', 'domain_sizex', 'domain_sizey', 'f_padding', 'seed']:
            args[key] = int(line[:-1])
        elif key in ['data_mult', 'ALPHA']:
            args[key] = float(line[:-1])
        else:
            args[key] = line[:-1]

    return Namespace(**args)

def setup_plot_data(data, pml, src):
    colored_yee = np.zeros((data.shape[0], data.shape[1],3))
    air_color = np.array([249,232,215], dtype=np.uint8)
    pml_color = np.array([255, 185, 0], dtype=np.uint8)
    src_color = np.array([52, 181, 168], dtype=np.uint8)

    top_mat_color = np.array([30, 30, 30], dtype=np.uint8)

    data = np.asarray(data)
    colored_yee = air_color+((data[:,:,None]-1)/15*(top_mat_color.astype(np.float32)-air_color.astype(np.float32))).astype(np.uint8)

    colored_yee = (pml[:,:,None]>0.5)*pml_color + (pml[:,:,None]<0.5)*colored_yee 

    thickened_src = np.abs(src[:,:,None])>1e-5
    thickened_src = thickened_src + np.roll(thickened_src, 1, axis=0) + np.roll(thickened_src, -1, axis=0) + \
                                    np.roll(thickened_src, 2, axis=0) + np.roll(thickened_src, -2, axis=0) + \
                                    np.roll(thickened_src, 3, axis=0) + np.roll(thickened_src, -3, axis=0) + \
                                    np.roll(thickened_src, 1, axis=1) + np.roll(thickened_src, -1, axis=1) + \
                                    np.roll(thickened_src, 2, axis=1) + np.roll(thickened_src, -2, axis=1) + \
                                    np.roll(thickened_src, 3, axis=1) + np.roll(thickened_src, -3, axis=1)

    colored_yee = (thickened_src>1e-5)*src_color + (np.abs(src[:,:,None])<1e-5)*colored_yee 

    return colored_yee

@jax.jit
def combine_bc(top_bc_train,bottom_bc_train,left_bc_train,right_bc_train):
    top_bc_train_RI = jnp.stack((jnp.real(top_bc_train), jnp.imag(top_bc_train)), axis=3)
    bottom_bc_train_RI = jnp.stack((jnp.real(bottom_bc_train), jnp.imag(bottom_bc_train)), axis=3)
    left_bc_train_RI = jnp.stack((jnp.real(left_bc_train), jnp.imag(left_bc_train)), axis=3)
    right_bc_train_RI = jnp.stack((jnp.real(right_bc_train), jnp.imag(right_bc_train)), axis=3)

    return jax.lax.stop_gradient(top_bc_train_RI), jax.lax.stop_gradient(bottom_bc_train_RI), jax.lax.stop_gradient(left_bc_train_RI), jax.lax.stop_gradient(right_bc_train_RI)

@eqx.filter_jit
def mat_model_eval(mat_model, mat_idx, yeex_batch_train, yeey_batch_train, top_bc_train_RI, bottom_bc_train_RI, left_bc_train_RI, right_bc_train_RI):
    return eqx.filter_vmap(mat_model)(yeex_batch_train, yeey_batch_train, top_bc_train_RI[mat_idx,:,:,:], bottom_bc_train_RI[mat_idx,:,:,:], left_bc_train_RI[mat_idx,:,:,:], right_bc_train_RI[mat_idx,:,:,:])

@eqx.filter_jit
def src_model_eval(src_model, src_idx, source_batch_train_RI, top_bc_train_RI, bottom_bc_train_RI, left_bc_train_RI, right_bc_train_RI):
    return eqx.filter_vmap(src_model)(source_batch_train_RI, top_bc_train_RI[src_idx,:,:,:], bottom_bc_train_RI[src_idx,:,:,:], left_bc_train_RI[src_idx,:,:,:], right_bc_train_RI[src_idx,:,:,:])

@eqx.filter_jit
def pml_model_eval(pml_model, pml_idx, Sx_f_batch_train_RI, Sy_f_batch_train_RI, top_bc_train_RI, bottom_bc_train_RI, left_bc_train_RI, right_bc_train_RI):
    return eqx.filter_vmap(pml_model)(Sx_f_batch_train_RI, Sy_f_batch_train_RI, top_bc_train_RI[pml_idx,:,:,:], bottom_bc_train_RI[pml_idx,:,:,:], left_bc_train_RI[pml_idx,:,:,:], right_bc_train_RI[pml_idx,:,:,:])

@partial(jax.jit, static_argnums=[0])
def momentum_bc_update(momentum, top_bc_train, bottom_bc_train, left_bc_train, right_bc_train, new_top_bc_train, new_bottom_bc_train, new_left_bc_train, new_right_bc_train):
    return momentum*top_bc_train + (1-momentum)*new_top_bc_train, \
           momentum*bottom_bc_train + (1-momentum)*new_bottom_bc_train, \
           momentum*left_bc_train + (1-momentum)*new_left_bc_train, \
           momentum*right_bc_train + (1-momentum)*new_right_bc_train

def plot_helper(data,title,path):
    plt.figure()
    ax = plt.gca()
    im = ax.imshow(data)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.savefig(path, transparent=True)
    plt.close()

@eqx.filter_jit
def stop_gradient(model):
    array_model, rest = eqx.partition(model, eqx.is_array)
    array_model = jax.lax.stop_gradient(array_model)
    return eqx.combine(array_model, rest)

@partial(jax.jit, static_argnums=[3,4,5,6,7,8])
def prepare_batched_data(DDM_img, DDM_Hy, DDM_source, model_bs, x_patches, y_patches, domain_sizex, domain_sizey, overlap_pixels):
    # pad img, source, pml and Hy to be size_x by size_y
    yeex_batch_train = [(1/2*(DDM_img+jnp.roll(DDM_img,1,axis=0)))[0+i*(domain_sizex-overlap_pixels) : 0+domain_sizex+i*(domain_sizex-overlap_pixels),\
                                                                  0+j*(domain_sizey-overlap_pixels) : 0+domain_sizey+j*(domain_sizey-overlap_pixels)] \
                        for i in range(x_patches) for j in range(y_patches)]

    yeex_batch_train = jnp.stack(yeex_batch_train).reshape(model_bs,domain_sizex,domain_sizey)

    yeey_batch_train = [1/2*(DDM_img+jnp.roll(DDM_img,1,axis=1))[0+i*(domain_sizex-overlap_pixels) : 0+domain_sizex+i*(domain_sizex-overlap_pixels),\
                                                                0+j*(domain_sizey-overlap_pixels) : 0+domain_sizey+j*(domain_sizey-overlap_pixels)] \
                        for i in range(x_patches) for j in range(y_patches)]

    yeey_batch_train = jnp.stack(yeey_batch_train).reshape(model_bs,domain_sizex,domain_sizey)

    y_batch_train = [DDM_Hy[i*(domain_sizex-overlap_pixels):domain_sizex+i*(domain_sizex-overlap_pixels),\
                            j*(domain_sizey-overlap_pixels):domain_sizey+j*(domain_sizey-overlap_pixels)]  for i in range(x_patches) for j in range(y_patches)]
    y_batch_train = jnp.stack(y_batch_train).reshape(model_bs,domain_sizex,domain_sizex)
    y_batch_train_RI = jnp.stack((jnp.real(y_batch_train), jnp.imag(y_batch_train)), axis=3)

    # batched sources and pmls:
    source_batch_train = [DDM_source[i*(domain_sizex-overlap_pixels):domain_sizex+i*(domain_sizex-overlap_pixels),\
                                     j*(domain_sizey-overlap_pixels):domain_sizey+j*(domain_sizey-overlap_pixels)]  for i in range(x_patches) for j in range(y_patches)]
    source_batch_train = jnp.stack(source_batch_train).reshape(model_bs,domain_sizex,domain_sizey)
    source_batch_train_RI = jnp.stack((jnp.real(source_batch_train), jnp.imag(source_batch_train)), axis=3)

    return yeex_batch_train, yeey_batch_train, y_batch_train_RI, source_batch_train_RI

@partial(jax.jit, static_argnums=[3,4,5,6])
def prepare_loaded_data(DDM_img, DDM_Hy, DDM_source, data_mult, dL, wl, overlap_pixels, EPSILON_0):
    DDM_img = DDM_img[0,0,:,:]
    DDM_Hy = data_mult*(DDM_Hy[0,0,:,:]+1j*DDM_Hy[0,1,:,:])
    DDM_source = data_mult*1j*2*jnp.pi*C_0*dL**2/wl*EPSILON_0*(DDM_source[0,0,:,:] + 1j*DDM_source[0,1,:,:])
    
    DDM_img = pad_periodic(DDM_img, Nx=overlap_pixels, Ny=overlap_pixels)
    DDM_Hy = pad_periodic(DDM_Hy, Nx=overlap_pixels, Ny=overlap_pixels)
    DDM_source = pad_periodic(DDM_source, Nx=overlap_pixels, Ny=overlap_pixels)

    return DDM_img, DDM_Hy, DDM_source

def main(args):
    key = jax.random.PRNGKey(args.seed)

    jax_devices = jax.devices('gpu')
    print("jax_devices: ", jax_devices)
    
    all_models = os.listdir(args.model_saving_path)

    # load the material model with JAX Equinox:
    matching_models = [i for i in all_models if i[:len(args.mat_model_name)] == args.mat_model_name]
    if len(matching_models) == 0:
        raise ValueError("no model found for ", args.mat_model_name)
    if len(matching_models)>1:
        raise ValueError("more than 1 models found!", matching_models)
    path = args.model_saving_path +matching_models[0]+"/best_model.eqx"
    with open(args.model_saving_path +matching_models[0]+'/config.txt') as f:
        mat_args = reconstruct_args(f)
    print("mat_args: ", mat_args)
    model_original = SM_FNO_mat(mat_args, key = key)
    mat_model = eqx.tree_deserialise_leaves(path, model_original)
    print("material model loaded from ", path)

    # load the src model:
    matching_models = [i for i in all_models if i[:len(args.src_model_name)] == args.src_model_name]
    if len(matching_models) == 0:
        raise ValueError("no model found for ", args.src_model_name)
    if len(matching_models)>1:
        raise ValueError("more than 1 models found!", matching_models)
    path = args.model_saving_path +matching_models[0]+"/best_model.eqx"
    with open(args.model_saving_path +matching_models[0]+'/config.txt') as f:
        src_args = reconstruct_args(f)
    print("src_args: ", src_args)
    model_original = SM_FNO_src(src_args, key = key)
    src_model = eqx.tree_deserialise_leaves(path, model_original)   
    print("source model loaded from ", path)

    # load the pml model:
    matching_models = [i for i in all_models if i[:len(args.pml_model_name)] == args.pml_model_name]
    if len(matching_models) == 0:
        raise ValueError("no model found for ", args.pml_model_name)
    if len(matching_models)>1:
        raise ValueError("more than 1 models found!", matching_models)
    path = args.model_saving_path +matching_models[0]+"/best_model.eqx"
    with open(args.model_saving_path +matching_models[0]+'/config.txt') as f:
        pml_args = reconstruct_args(f)
    print("pml_args: ", pml_args)
    model_original = SM_FNO_pml(pml_args, key = key)
    pml_model = eqx.tree_deserialise_leaves(path, model_original)
    print("pml model loaded from ", path)
    
    ds = DDM_Dataset(args.data_folder, total_sample_number = args.total_sample_number, data_type=np.float32 if args.float_bits==32 else np.float64)
    torch.manual_seed(42)
    DDM_loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    omega = 2 * jnp.pi * C_0 / args.wl

    total_shape = args.total_shapex, args.total_shapey
    assert total_shape[0] % (args.domain_sizex-args.overlap_pixels) == 0
    assert total_shape[1] % (args.domain_sizey-args.overlap_pixels) == 0
    x_patches = total_shape[0]//(args.domain_sizex-args.overlap_pixels)
    y_patches = total_shape[1]//(args.domain_sizey-args.overlap_pixels)
    print("x_patches: ", x_patches)
    print("y_patches: ", y_patches)
    model_bs = x_patches*y_patches

    size_x = args.domain_sizex+(x_patches-1)*(args.domain_sizex-args.overlap_pixels)
    size_y = args.domain_sizey+(y_patches-1)*(args.domain_sizey-args.overlap_pixels)
    print("size: ", size_x, size_y, total_shape)
    # pml is the same for all subdomains:

    pml_map = jnp.zeros(total_shape, dtype=np.float32)
    pml_map = pml_map.at[:args.pml_thickness+1,:].set(1)
    pml_map = pml_map.at[-args.pml_thickness+1:,:].set(1)

    if not args.periodic:
        pml_map = pml_map.at[:,:args.pml_thickness+1].set(1)
        pml_map = pml_map.at[:,-args.pml_thickness+1:].set(1)

    Sx_2D_f, Sy_2D_f = make_Sx_Sy(omega, args.dL, total_shape[0], args.pml_thickness, total_shape[1], args.pml_thickness if not args.periodic else 0, _dir='f', float_bits=args.float_bits)
    Sx_2D_b, Sy_2D_b = make_Sx_Sy(omega, args.dL, total_shape[0], args.pml_thickness, total_shape[1], args.pml_thickness if not args.periodic else 0, _dir='b', float_bits=args.float_bits)

    pml_map = pad_periodic(pml_map, Nx=args.overlap_pixels, Ny=args.overlap_pixels)
    Sx_2D_f = pad_periodic(Sx_2D_f, Nx=args.overlap_pixels, Ny=args.overlap_pixels)
    Sy_2D_f = pad_periodic(Sy_2D_f, Nx=args.overlap_pixels, Ny=args.overlap_pixels)
    Sx_2D_b = pad_periodic(Sx_2D_b, Nx=args.overlap_pixels, Ny=args.overlap_pixels)
    Sy_2D_b = pad_periodic(Sy_2D_b, Nx=args.overlap_pixels, Ny=args.overlap_pixels)

    pml_batch_train = [pml_map[args.starting_x+i*(args.domain_sizex-args.overlap_pixels):args.starting_x+args.domain_sizex+i*(args.domain_sizex-args.overlap_pixels),\
                               args.starting_y+j*(args.domain_sizey-args.overlap_pixels):args.starting_y+args.domain_sizey+j*(args.domain_sizey-args.overlap_pixels)]  for i in range(x_patches) for j in range(y_patches)]
    pml_batch_train = jnp.stack(pml_batch_train).reshape(model_bs,args.domain_sizex,args.domain_sizey)

    Sx_f_batch_train = jnp.stack([Sx_2D_f[args.starting_x+i*(args.domain_sizex-args.overlap_pixels):args.starting_x+args.domain_sizex+i*(args.domain_sizex-args.overlap_pixels),\
                                  args.starting_y+j*(args.domain_sizey-args.overlap_pixels):args.starting_y+args.domain_sizey+j*(args.domain_sizey-args.overlap_pixels)]  for i in range(x_patches) for j in range(y_patches)]).reshape(model_bs,args.domain_sizex,args.domain_sizex)
    Sx_b_batch_train = jnp.stack([Sx_2D_b[args.starting_x+i*(args.domain_sizex-args.overlap_pixels):args.starting_x+args.domain_sizex+i*(args.domain_sizex-args.overlap_pixels),\
                                  args.starting_y+j*(args.domain_sizey-args.overlap_pixels):args.starting_y+args.domain_sizey+j*(args.domain_sizey-args.overlap_pixels)]  for i in range(x_patches) for j in range(y_patches)]).reshape(model_bs,args.domain_sizex,args.domain_sizex)

    Sy_f_batch_train = jnp.stack([Sy_2D_f[args.starting_x+i*(args.domain_sizex-args.overlap_pixels):args.starting_x+args.domain_sizex+i*(args.domain_sizex-args.overlap_pixels),\
                                  args.starting_y+j*(args.domain_sizey-args.overlap_pixels):args.starting_y+args.domain_sizey+j*(args.domain_sizey-args.overlap_pixels)]  for i in range(x_patches) for j in range(y_patches)]).reshape(model_bs,args.domain_sizex,args.domain_sizex)
    Sy_b_batch_train = jnp.stack([Sy_2D_b[args.starting_x+i*(args.domain_sizex-args.overlap_pixels):args.starting_x+args.domain_sizex+i*(args.domain_sizex-args.overlap_pixels),\
                                  args.starting_y+j*(args.domain_sizey-args.overlap_pixels):args.starting_y+args.domain_sizey+j*(args.domain_sizey-args.overlap_pixels)]  for i in range(x_patches) for j in range(y_patches)]).reshape(model_bs,args.domain_sizex,args.domain_sizex)

    Sx_f_batch_train_RI = jnp.stack((jnp.real(Sx_f_batch_train), jnp.imag(Sx_f_batch_train)), axis=3)
    Sy_f_batch_train_RI = jnp.stack((jnp.real(Sy_f_batch_train), jnp.imag(Sy_f_batch_train)), axis=3)

    convergence_data = []
    for sample_id, sample_batched in enumerate(DDM_loader):
        if sample_id == 6:
            break

        this_converge = []
        this_data = {}

        this_converge = []
        this_data = {}

        DDM_img, DDM_Hy, DDM_source = prepare_loaded_data(jnp.asarray(sample_batched['structure']), jnp.asarray(sample_batched['field']), jnp.asarray(sample_batched['source']), args.data_mult, args.dL, args.wl, args.overlap_pixels, EPSILON_0)
        yeex_batch_train, yeey_batch_train, y_batch_train_RI, source_batch_train_RI = prepare_batched_data(DDM_img, DDM_Hy, DDM_source, model_bs, x_patches, y_patches, args.domain_sizex, args.domain_sizey, args.overlap_pixels)
        
        # zero robin b.c. init.
        top_bc_train, bottom_bc_train, left_bc_train, right_bc_train = init_zero_robin_bc(args, x_patches, y_patches)

        colored_setup = setup_plot_data(DDM_img[args.starting_x:args.starting_x+size_x, args.starting_y:args.starting_y+size_y], \
                                        pml_map[args.starting_x:args.starting_x+size_x, args.starting_y:args.starting_y+size_y],\
                                        DDM_source[args.starting_x:args.starting_x+size_x, args.starting_y:args.starting_y+size_y] )

        # get the indices of the batched data for each model:
        if sample_id == 0:
            src_idx = get_non_zero_indices(source_batch_train_RI)
            pml_idx = get_non_zero_indices(pml_batch_train)
            assert len(set(src_idx).intersection(set(pml_idx))) == 0
            mat_idx = tuple(sorted(list(set(range(source_batch_train_RI.shape[0]))-set(src_idx)-set(pml_idx))))
            print(f"batch size for mat,src,pml: {len(mat_idx)}, {len(src_idx)}, {len(pml_idx)}")

        # index all fixed data:
        yeex_batch_train_indexed = yeex_batch_train[mat_idx,:,:]
        yeey_batch_train_indexed = yeey_batch_train[mat_idx,:,:]
        source_batch_train_RI_indexed = source_batch_train_RI[src_idx,:,:]
        Sx_f_batch_train_RI_indexed = Sx_f_batch_train_RI[pml_idx,:,:,1]
        Sy_f_batch_train_RI_indexed = Sy_f_batch_train_RI[pml_idx,:,:,1]

        this_vmax_r = jnp.max(jnp.real(DDM_Hy[args.starting_x:args.starting_x+size_x, args.starting_y:args.starting_y+size_y]))
        this_vmin_r = jnp.min(jnp.real(DDM_Hy[args.starting_x:args.starting_x+size_x, args.starting_y:args.starting_y+size_y]))

        this_vmax_i = jnp.max(jnp.imag(DDM_Hy[args.starting_x:args.starting_x+size_x, args.starting_y:args.starting_y+size_y]))
        this_vmin_i = jnp.min(jnp.imag(DDM_Hy[args.starting_x:args.starting_x+size_x, args.starting_y:args.starting_y+size_y]))

        for k in range(args.DDM_iters):
            time3 = time.time()
            if (k+1)%10==0:
                print(k+1)
            top_bc_train_RI, bottom_bc_train_RI, left_bc_train_RI, right_bc_train_RI = combine_bc(top_bc_train, bottom_bc_train, left_bc_train, right_bc_train)

            time4 = time.time()
            mat_logits_RI, src_logits_RI, pml_logits_RI=None,None,None
            if len(mat_idx)>0:
                mat_logits_RI  = mat_model_eval(mat_model, mat_idx, yeex_batch_train_indexed, yeey_batch_train_indexed, top_bc_train_RI, bottom_bc_train_RI, left_bc_train_RI, right_bc_train_RI)

            if len(src_idx)>0:
                src_logits_RI  = src_model_eval(src_model, src_idx, args.source_mult*source_batch_train_RI_indexed, top_bc_train_RI, bottom_bc_train_RI, left_bc_train_RI, right_bc_train_RI)

            if len(pml_idx)>0:
                pml_logits_RI  = pml_model_eval(pml_model, pml_idx, Sx_f_batch_train_RI_indexed, Sy_f_batch_train_RI_indexed, top_bc_train_RI, bottom_bc_train_RI, left_bc_train_RI, right_bc_train_RI)

            time45 = time.time()
            logits = combine_batch(x_patches*y_patches, mat_idx, src_idx, pml_idx, y_batch_train_RI.shape, mat_logits_RI, src_logits_RI, pml_logits_RI)

            time5 = time.time()

            plt.rcParams["font.size"] = '14'
            plt.rcParams["font.family"] = 'Times New Roman'

            if (k+1)%args.div_k==0:
                # reconstruct the whole field
                intermediate_result = reconstruct_complex(logits, x_patches=x_patches, y_patches=y_patches, d_sx=args.domain_sizex, d_sy=args.domain_sizey, ol=args.overlap_pixels)
                this_Hy = DDM_Hy[args.starting_x:args.starting_x+size_x, args.starting_y:args.starting_y+size_y]
                
                loss = jnp.mean(jnp.abs(intermediate_result - this_Hy))/ \
                       jnp.mean(jnp.abs(this_Hy))

                if args.write_video:
                    fig, axs = plt.subplots(3)
                    for a in axs.flatten():
                        a.set_xticks([])
                        a.set_yticks([])
                    axs[0].imshow(colored_setup)
                    im = axs[1].imshow(jnp.real(this_Hy), cmap='seismic')
                    im = axs[2].imshow(jnp.real(intermediate_result), cmap='seismic', vmax=this_vmax_r, vmin=this_vmin_r)
                    annotate_x = 0.5
                    annotate_y = -0.15
                    annotate_content = f"Iteration: {(k+1):d}"
                    plt.annotate(annotate_content, (annotate_x, annotate_y), xycoords="axes fraction", ha="center", fontsize=10)

                    annotate_x = 0.5
                    annotate_y = -0.3
                    annotate_content = f"rel. L1 loss: {loss:.3f}"
                    plt.annotate(annotate_content, (annotate_x, annotate_y), xycoords="axes fraction", ha="center", fontsize=10)


                    plt.savefig(f'frames/frame_{k:04d}.png', bbox_inches='tight', transparent=True, dpi=300)
                    plt.close()

                this_converge.append(loss)

            time6 = time.time()

            # Then prepare the data for next iteration:
            new_top_bc_train, new_bottom_bc_train, new_left_bc_train, new_right_bc_train = new_iter_bcs_periodic_average_bc(logits, yeex_batch_train, yeey_batch_train,  x_patches, y_patches, domain_sizex=args.domain_sizex, domain_sizey=args.domain_sizey, overlap_pixels=args.overlap_pixels)

            top_bc_train, bottom_bc_train, left_bc_train, right_bc_train = momentum_bc_update(args.momentum, top_bc_train, bottom_bc_train, left_bc_train, right_bc_train, new_top_bc_train, new_bottom_bc_train, new_left_bc_train, new_right_bc_train)
            
            time7 = time.time()

            if k<5:
                print(f"model inference time: {time45-time4}, combine_time:{time5-time45} real_complex conversion time: {time4-time3}, \
                        plot time: {time6-time5}, update next time: {time7-time6}")
                print(f"total step time: {time7-time3}")


        if args.write_video:
            video_filename = f'video_{sample_id}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 video
            fps = 5  # Frames per second
            frame = cv2.imread(f'frames/frame_{(args.div_k-1):04d}.png')  # Load the first frame to get dimensions
            frame_height, frame_width, _ = frame.shape
            video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

            for i in range(args.div_k,args.DDM_iters+1,args.div_k):
                filename = f'frames/frame_{(i-1):04d}.png'
                frame = cv2.imread(filename)
                video_writer.write(frame)

            video_writer.release()

        convergence_data.append(this_converge)

    convergence_data = np.array(convergence_data)
    iters = args.div_k*np.array(range(1, convergence_data.shape[1]+1))
    plt.figure()
    plt.plot(iters, convergence_data.T)
    domain_size = args.data_folder.split('_')[-3]
    plt.title("subdomain grid: %d x %d, mean loss: %.4f" % (x_patches, y_patches, np.mean(convergence_data[:,-1])))
    plt.ylim(0,0.6)
    plt.xlabel("Iterations")
    plt.ylabel("Relative L1 loss")
    plt.savefig("eval_convergence.png", transparent=True, dpi=300)
    plt.close()

    print(convergence_data.shape)
    mean_error_each_iter = np.mean(convergence_data, axis=0)
    print([(i, mean_error_each_iter[i]) for i in range(len(mean_error_each_iter))])

if __name__ == '__main__': 
    main(args)
