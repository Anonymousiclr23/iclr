# import  torch, os
# import torch.nn.functional as F
import  jax.numpy as jnp
from functools import partial
from matplotlib import pyplot as plt
import jax

EPSILON_0 = 8.85418782e-12        # vacuum permittivity
MU_0 = 1.25663706e-6              # vacuum permeability
C_0 = 1 / (EPSILON_0 * MU_0)**.5  # speed of light in vacuum
ETA_0 = (MU_0 / EPSILON_0)**.5    # vacuum impedance
Q_e = 1.602176634e-19             # funamental charge

def plot_helper(data,step,path):
    plt.figure()
    ax = plt.gca()
    im = ax.imshow(data.detach().cpu().numpy())
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.title(f"epoch{step}")
    plt.savefig(path, transparent=True)
    plt.close()

def subplots_helper(data, titles, path, shape, clims=None):
    rows, cols = shape
    plt.rcParams["font.size"] = "10"
    fig, ax = plt.subplots(rows, cols)
    for r in range(rows):
        for c in range(cols):
            idx = r*cols + c
            im = ax[r,c].imshow(np.abs(data[idx].detach().cpu().numpy()))
            ax[r,c].set_title(titles[idx], fontsize=6)
            if clims is not None:
                im.set_clim(clims[idx])
            plt.colorbar(im, ax=ax[r,c])

    plt.savefig(path, dpi=500)

def init_zero_robin_bc(args, x_patches=None, y_patches=None, device='cpu'):
    if x_patches is None:
        x_patches = args.x_patches
    if y_patches is None:
        y_patches = args.y_patches
    top_bc = jnp.zeros((x_patches*y_patches, 1, args.domain_sizey), dtype=jnp.complex64)
    bottom_bc = jnp.zeros((x_patches*y_patches, 1, args.domain_sizey), dtype=jnp.complex64)
    left_bc = jnp.zeros((x_patches*y_patches, args.domain_sizex, 1), dtype=jnp.complex64)
    right_bc = jnp.zeros((x_patches*y_patches, args.domain_sizex, 1), dtype=jnp.complex64)

    return top_bc, bottom_bc, left_bc, right_bc

@partial(jax.jit, static_argnums=[1,2,3,4,5])
def reconstruct_complex(logits, x_patches, y_patches, d_sx=64, d_sy=64, ol=16, exclude_corners = False):

    patched_logits = logits.reshape(x_patches, y_patches, d_sx, d_sy)

    size_x = d_sx+(x_patches-1)*(d_sx-ol)
    size_y = d_sy+(y_patches-1)*(d_sy-ol)
    
    reconstructed = jnp.zeros((size_x, size_y), dtype = jnp.complex64)

    for i in range(x_patches):
        for j in range(y_patches):
            reconstructed = reconstructed.at[i*(d_sx-ol):i*(d_sx-ol)+d_sx,j*(d_sy-ol):j*(d_sy-ol)+d_sy].set(reconstructed[i*(d_sx-ol):i*(d_sx-ol)+d_sx,j*(d_sy-ol):j*(d_sy-ol)+d_sy]+patched_logits[i,j,:,:])

    # for the double counted pixels, divide by 2, 
    for i in range(x_patches-1):
        reconstructed = reconstructed.at[(i+1)*(d_sx-ol):(i+1)*(d_sx-ol)+ol, :].set(reconstructed[(i+1)*(d_sx-ol):(i+1)*(d_sx-ol)+ol, :]/2)
        if exclude_corners:
            corner_y_coords = [0,-1] + [(k+1)*(d_sy-ol) for k in range(y_patches - 1)] + [(k+1)*(d_sy-ol)+ol-1 for k in range(y_patches - 1)]
            for y_coord in corner_y_coords:
                reconstructed = reconstructed.at[(i+1)*(d_sx-ol), y_coord].set(2*reconstructed[(i+1)*(d_sx-ol), y_coord])
                reconstructed = reconstructed.at[(i+1)*(d_sx-ol)+ol-1, y_coord].set(2*reconstructed[(i+1)*(d_sx-ol)+ol-1, y_coord])

    for j in range(y_patches-1):
        reconstructed = reconstructed.at[:, (j+1)*(d_sy-ol):(j+1)*(d_sy-ol)+ol].set(reconstructed[:, (j+1)*(d_sy-ol):(j+1)*(d_sy-ol)+ol]/2)
        if exclude_corners:
            corner_x_coords = [0,-1] + [(k+1)*(d_sx-ol) for k in range(x_patches - 1)] + [(k+1)*(d_sx-ol)+ol-1 for k in range(x_patches - 1)]
            for x_coord in corner_x_coords:
                reconstructed = reconstructed.at[x_coord, (j+1)*(d_sy-ol)].set(2*reconstructed[x_coord, (j+1)*(d_sy-ol)])
                reconstructed = reconstructed.at[x_coord, (j+1)*(d_sy-ol)+ol-1].set(2*reconstructed[x_coord, (j+1)*(d_sy-ol)+ol-1])

    return reconstructed

@partial(jax.jit, static_argnums=[3,4,6,7,8])
def new_iter_bcs_periodic_average_bc(logits, yeex_batch_train, yeey_batch_train, x_patches, y_patches, bloch_phases=(0,0), domain_sizex=64, domain_sizey=64, overlap_pixels=16):
    patched_logits = logits.reshape(x_patches, y_patches, domain_sizex,domain_sizey)
    patched_yeex = yeex_batch_train.reshape(x_patches, y_patches, domain_sizex,domain_sizey)
    patched_yeey = yeey_batch_train.reshape(x_patches, y_patches, domain_sizex,domain_sizey)

    top_bc = jnp.zeros((x_patches, y_patches, 1, domain_sizey), dtype=jnp.complex64)
    bottom_bc = jnp.zeros((x_patches, y_patches, 1, domain_sizey), dtype=jnp.complex64)
    left_bc = jnp.zeros((x_patches, y_patches, domain_sizex, 1), dtype=jnp.complex64)
    right_bc = jnp.zeros((x_patches, y_patches, domain_sizex, 1), dtype=jnp.complex64)

    # top bc
    yee = patched_yeex[:,:,1:2,:]
    roll = jnp.roll(patched_logits, 1, axis=0)
    bc = 1/2*(roll[:,:,domain_sizex-overlap_pixels  :domain_sizex-overlap_pixels+1,:]+\
              roll[:,:,domain_sizex-overlap_pixels+1:domain_sizex-overlap_pixels+2,:])
    d_w = (roll[:,:,domain_sizex-overlap_pixels  :domain_sizex-overlap_pixels+1,:]-\
           roll[:,:,domain_sizex-overlap_pixels+1:domain_sizex-overlap_pixels+2,:])
    bc = bc.at[0].set(bc[0]*jnp.exp(-1j*bloch_phases[0]))
    d_w = d_w.at[0].set(d_w[0]*jnp.exp(-1j*bloch_phases[0]))
    top_bc = robin(yee, bc, d_w=d_w)

    # bottom bc
    yee = patched_yeex[:,:,-1:,:]
    roll = jnp.roll(patched_logits, -1, axis=0)
    bc = 1/2*(roll[:,:,overlap_pixels-1:overlap_pixels  ,:]+\
              roll[:,:,overlap_pixels-2:overlap_pixels-1,:])

    d_w = (roll[:,:,overlap_pixels-1:overlap_pixels  ,:] - \
           roll[:,:,overlap_pixels-2:overlap_pixels-1,:])
    bc = bc.at[-1].set(bc[-1]*jnp.exp(1j*bloch_phases[0]))
    d_w = d_w.at[-1].set(d_w[-1]*jnp.exp(1j*bloch_phases[0]))
    bottom_bc = robin(yee, bc, d_w=d_w)

    # left bc
    yee = patched_yeey[:,:,:,1:2]
    roll = jnp.roll(patched_logits, 1, axis=1)
    bc = 1/2*(roll[:,:,:,domain_sizey-overlap_pixels  :domain_sizey-overlap_pixels+1]+\
              roll[:,:,:,domain_sizey-overlap_pixels+1:domain_sizey-overlap_pixels+2])
    d_w = (roll[:,:,:,domain_sizey-overlap_pixels  :domain_sizey-overlap_pixels+1] - \
           roll[:,:,:,domain_sizey-overlap_pixels+1:domain_sizey-overlap_pixels+2])
    bc = bc.at[:,0].set(bc[:,0]*jnp.exp(-1j*bloch_phases[1]))
    d_w = d_w.at[:,0].set(d_w[:,0]*jnp.exp(-1j*bloch_phases[1]))
    left_bc = robin(yee, bc, d_w=d_w)

    # right bc
    yee = patched_yeey[:,:,:,-1:]
    roll = jnp.roll(patched_logits, -1, axis=1)
    bc = 1/2*(roll[:,:,:,overlap_pixels-1:overlap_pixels  ]+\
              roll[:,:,:,overlap_pixels-2:overlap_pixels-1])
    d_w = (roll[:,:,:,overlap_pixels-1:overlap_pixels  ] - \
           roll[:,:,:,overlap_pixels-2:overlap_pixels-1])
    bc = bc.at[:,-1].set(bc[:,-1]*jnp.exp(1j*bloch_phases[1]))
    d_w = d_w.at[:,-1].set(d_w[:,-1]*jnp.exp(1j*bloch_phases[1]))
    right_bc = robin(yee, bc, d_w=d_w)

    return top_bc.reshape((x_patches*y_patches, 1, domain_sizey)), \
           bottom_bc.reshape((x_patches*y_patches, 1, domain_sizey)), \
           left_bc.reshape((x_patches*y_patches, domain_sizex, 1)), \
           right_bc.reshape((x_patches*y_patches, domain_sizex, 1))

def dirichlet(bc, d_v, d_w, args):
    # bc: the boundary field to be transform
    # d_v: the derivative of fields in v
    # d_w: the derivative of fields in w
    return bc

@jax.jit
def robin(yee, bc, d_v=None, d_w=None, wl=1050e-9, dL=6.25e-9):
    # bc: the boundary field to be transform
    # d_v: the derivative of fields in v
    # d_w: the derivative of fields in w
    g = 1j*2*jnp.pi*jnp.sqrt(yee)/wl*dL*bc+d_w
    return g


@partial(jax.jit, static_argnums=[1,2])
def pad_periodic(arr, Nx=0, Ny=0, bloch_phases = None):
    # arr size: (sx, sy)
    if bloch_phases is not None:
        bp_x = bloch_phases[0]
        bp_y = bloch_phases[1]
        arr = jnp.concatenate((arr, jnp.exp(1j*bp_x)*arr[:Nx,:]), axis=0)
        arr = jnp.concatenate((arr, jnp.exp(1j*bp_y)*arr[:,:Ny]), axis=1)
        return arr
    else:
        return jnp.pad(arr, ((0,Nx),(0,Ny)), mode="wrap")

def make_Sx_Sy(omega, dL, Nx, Nx_pml, Ny, Ny_pml, _dir="f", float_bits=None):
    dtype = jnp.complex64 if (float_bits==32) else jnp.complex128
    if _dir == 'f':
        sfactor_x = create_sfactor_f(omega, dL, Nx, Nx_pml, float_bits=float_bits)
        sfactor_y = create_sfactor_f(omega, dL, Ny, Ny_pml, float_bits=float_bits)
    elif _dir == 'b':
        sfactor_x = create_sfactor_b(omega, dL, Nx, Nx_pml, float_bits=float_bits)
        sfactor_y = create_sfactor_b(omega, dL, Ny, Ny_pml, float_bits=float_bits)

    Sx_2D = jnp.zeros((Nx,Ny), dtype=dtype)
    Sy_2D = jnp.zeros((Nx,Ny), dtype=dtype)

    for i in range(0, Ny):
        Sx_2D = Sx_2D.at[:, i].set(sfactor_x)
    for i in range(0, Nx):
        Sy_2D = Sy_2D.at[i, :].set(sfactor_y)

    return Sx_2D, Sy_2D

def create_sfactor_f(omega, dL, N, N_pml, float_bits=None):
    # forward
    dtype = jnp.complex64 if (float_bits==32) else jnp.complex128
    sfactor_array = jnp.ones(N, dtype=dtype)

    if N_pml == 0:
        return sfactor_array

    dw = N_pml*dL
    for i in range(N):
        if i <= N_pml:
            sfactor_array = sfactor_array.at[i].set(s_value(dL * (N_pml - i + 0.5), dw, omega))
        elif i > N - N_pml:
            sfactor_array = sfactor_array.at[i].set(s_value(dL * (i - (N - N_pml) - 0.5), dw, omega))

    return sfactor_array


def create_sfactor_b(omega, dL, N, N_pml, float_bits=None):
    # backward
    dtype = jnp.complex64 if (float_bits==32) else jnp.complex128
    sfactor_array = jnp.ones(N, dtype=dtype)
    
    if N_pml == 0:
        return sfactor_array

    dw = N_pml*dL
    for i in range(N):
        if i <= N_pml:
            sfactor_array = sfactor_array.at[i].set(s_value(dL * (N_pml - i + 1), dw, omega))
        elif i > N - N_pml:
            sfactor_array = sfactor_array.at[i].set(s_value(dL * (i - (N - N_pml) - 1), dw, omega))

    return sfactor_array

def sig_w(l, dw, m=3, lnR=-30):
    """ Fictional conductivity, note that these values might need tuning """
    sig_max = -(m + 1) * lnR / (2 * ETA_0 * dw)
    return sig_max * (l / dw)**m

def s_value(l, dw, omega):
    """ S-value to use in the S-matrices """
    # l is distance to the boundary of pml (close to the center)
    # dw is the physical thickness of pml (N_pml * dL)

    return 1 - 1j * sig_w(l, dw) / (omega * EPSILON_0)


