import math
from functools import partial

import jax
from jax import jit
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import optax  # https://github.com/deepmind/optax

import equinox as eqx
from jax import core

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
def Hz_to_Ex(Hz_R, Hz_I, dL, omega, yeex, EPSILON_0 = EPSILON_0):
    # print("sh1: ", Hz_R.shape, yeex.shape)
    # x = 1 / 2 * (eps_grid[:, :, 1:, :] + eps_grid[:, :, 0:-1, :]) # Material averaging
    Ex_R = -(Hz_I[:, 1:, 1:-1] - Hz_I[:, 0:-1, 1:-1])/dL/omega/EPSILON_0/yeex[:, :-1] # The returned Ex is corresponding to Ex_ceviche[:, 0:-1]
    Ex_I = (Hz_R[:, 1:, 1:-1] - Hz_R[:, 0:-1, 1:-1])/dL/omega/EPSILON_0/yeex[:, :-1]

    return jnp.stack((Ex_R, Ex_I), axis = 1)

@eqx.filter_jit
def Hz_to_Ey(Hz_R, Hz_I, dL, omega, yeey, EPSILON_0 = EPSILON_0):
    # print("sh2: ", Hz_R.shape, yeey.shape)
    # y = 1 / 2 * (eps_grid[:, :, 1:, :] + torch.roll(eps_grid[:, :, 1:, :], 1, dims = 3))
    Ey_R = (Hz_I[:,1:-1,1:] - Hz_I[:,1:-1,:-1])/dL/omega/EPSILON_0/yeey[:, :, :-1] # The returned Ey is corresponding to Ey_ceviche[0:-1, :]
    Ey_I = -(Hz_R[:,1:-1,1:] - Hz_R[:,1:-1,:-1])/dL/omega/EPSILON_0/yeey[:, :, :-1]
    return jnp.stack((Ey_R, Ey_I), axis = 1)

@eqx.filter_jit
def E_to_Hz(Ey_R, Ey_I, Ex_R, Ex_I, dL, omega, MU_0 = MU_0):
    # print("sh3:", Ey_R.shape, Ey_I.shape, Ex_R.shape, Ex_I.shape)
    Hz_R = ((Ey_I[:, :, 1:] - Ey_I[:, :, 0:-1]) - (Ex_I[:, 1:, :] - Ex_I[:, 0:-1, :]))/dL/omega/MU_0
    Hz_I = -((Ey_R[:, :, 1:] - Ey_R[:, :, 0:-1]) - (Ex_R[:, 1:, :] - Ex_R[:, 0:-1, :]))/dL/omega/MU_0
    return jnp.stack((Hz_R, Hz_I), axis = 1) # -Hy[1:, :]

@eqx.filter_jit
def H_to_H(Hz_R, Hz_I, dL, omega, yeex, yeey, EPSILON_0 = EPSILON_0, MU_0 = MU_0):
    FD_Ex = Hz_to_Ex(Hz_R, Hz_I, dL, omega, yeex, EPSILON_0)
    FD_Ey = Hz_to_Ey(Hz_R, Hz_I, dL, omega, yeey, EPSILON_0)
    FD_H = E_to_Hz(FD_Ey[:, 0], FD_Ey[:, 1], FD_Ex[:, 0], FD_Ex[:, 1], dL, omega, MU_0)
    return FD_H

class BasicBlock(eqx.Module):
    ALPHA: float
    expansion: int
    conv1: eqx.nn.Conv2d
    # bn1: eqx.experimental.BatchNorm
    conv2: eqx.nn.Conv2d
    # bn2: eqx.experimental.BatchNorm
    shortcut: eqx.nn.Identity
    def __init__(self, in_planes, planes, ALPHA, stride, *, key):
        super(BasicBlock, self).__init__()
        keys = jax.random.split(key, num=3)
        self.expansion = 1
        self.ALPHA = ALPHA
        self.conv1 = eqx.nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, use_bias=False, key=keys[0])
        # self.bn1 = eqx.experimental.BatchNorm(input_size=planes, axis_name="batch")
        self.conv2 = eqx.nn.Conv2d(planes, planes, kernel_size=3,stride=stride, padding=1, use_bias=False, key=keys[1])
        # self.bn2 = eqx.experimental.BatchNorm(input_size=planes, axis_name="batch")

        self.shortcut = eqx.nn.Identity()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = eqx.nn.Sequential(
                (eqx.nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, use_bias=False, key=keys[2]),
                # eqx.experimental.BatchNorm(input_size=self.expansion*planes, axis_name="batch")
                )
            )

    def __call__(self, x):
        # out = jax.nn.leaky_relu(self.bn1(self.conv1(x)), negative_slope=self.ALPHA)
        # out = self.bn2(self.conv2(out))
        out = jax.nn.leaky_relu(self.conv1(x), negative_slope=self.ALPHA)
        out = self.conv2(out)

        out += self.shortcut(x)
        out = jax.nn.leaky_relu(out, negative_slope=self.ALPHA)
        return out

class BasicBlock_without_shortcut(eqx.Module):
    ALPHA: float
    expansion: int
    conv1: eqx.nn.Conv2d
    # bn1: eqx.experimental.BatchNorm
    conv2: eqx.nn.Conv2d
    # bn2: eqx.experimental.BatchNorm
    # shortcut: eqx.nn.Identity
    def __init__(self, in_planes, planes, ALPHA, stride, *, key):
        super(BasicBlock_without_shortcut, self).__init__()
        keys = jax.random.split(key, num=3)
        self.expansion = 1
        self.ALPHA = ALPHA
        self.conv1 = eqx.nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, use_bias=False, key=keys[0])
        # self.bn1 = eqx.experimental.BatchNorm(input_size=planes, axis_name="batch")
        self.conv2 = eqx.nn.Conv2d(planes, planes, kernel_size=3,stride=stride, padding=1, use_bias=False, key=keys[1])
        # self.bn2 = eqx.experimental.BatchNorm(input_size=planes, axis_name="batch")

        # self.shortcut = eqx.nn.Identity()

        # if stride != 1 or in_planes != self.expansion*planes:
        #     self.shortcut = eqx.nn.Sequential(
        #         (eqx.nn.Conv2d(in_planes, self.expansion*planes,
        #                   kernel_size=1, stride=stride, use_bias=False, key=keys[2]),
        #         # eqx.experimental.BatchNorm(input_size=self.expansion*planes, axis_name="batch")
        #         )
        #     )

    def __call__(self, x):
        # out = jax.nn.leaky_relu(self.bn1(self.conv1(x)), negative_slope=self.ALPHA)
        # out = self.bn2(self.conv2(out))
        out = jax.nn.leaky_relu(self.conv1(x), negative_slope=self.ALPHA)
        out = self.conv2(out)

        # out += self.shortcut(x)
        # out = jax.nn.leaky_relu(out, negative_slope=self.ALPHA)
        return out

class Modulated_SpectralConv2d(eqx.Module):
    in_channels: int
    out_channels: int
    modes1: int
    modes2: int
    scale: float
    weights1: jax.numpy.ndarray
    weights2: jax.numpy.ndarray
    def __init__(self, in_channels, out_channels, modes1, modes2, *, key):
        super(Modulated_SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        wkey1, wkey2= jax.random.split(key, 2)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        # self.scale = 1

        # self.weights1 = self.scale * jax.random.normal(wkey1, (in_channels, out_channels, self.modes1, self.modes2), dtype=jnp.complex64)
        # self.weights2 = self.scale * jax.random.normal(wkey2, (in_channels, out_channels, self.modes1, self.modes2), dtype=jnp.complex64)

        self.weights1 = self.scale * jax.random.uniform(wkey1, (in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = self.scale * jax.random.uniform(wkey2, (in_channels, out_channels, self.modes1, self.modes2, 2))

    # Complex multiplication
    # @eqx.filter_jit
    def compl_mul2d(self, input, weights):
        # (in_channel, x,y ), (in_channel, out_channel, x,y) -> (out_channel, x,y)
        return jnp.einsum("ixy,ioxy->oxy", input, weights)

    # @eqx.filter_jit
    # @partial(jax.jit, static_argnames=['shape'])
    def __call__(self, shape, mod1, mod2, x):
        x_ft = jnp.fft.rfft2(x)

        # Multiply relevant Fourier modes
        # out_ft = jnp.zeros_like(x_ft)
        out_ft = jnp.zeros((shape[0], shape[1], shape[2]//2 + 1), dtype=jnp.complex64)

        # print("sh1: ", (self.weights1.shape, mod1.shape, (self.weights1*mod1).shape))

        complex_mult1 = jnp.zeros((self.out_channels,self.out_channels,self.modes1,self.modes2), dtype=jnp.complex64)
        complex_mult1 = complex_mult1.at[:,:,:,:].set(jnp.multiply(jax.lax.complex(self.weights1[...,0],self.weights1[...,1]),mod1))
        complex_mult2 = jnp.zeros((self.out_channels,self.out_channels,self.modes1,self.modes2), dtype=jnp.complex64)
        complex_mult2 = complex_mult2.at[:,:,:,:].set(jnp.multiply(jax.lax.complex(self.weights2[...,0],self.weights2[...,1]),mod2))
        
        y1 = self.compl_mul2d(x_ft[:, :self.modes1, :self.modes2], complex_mult1)
        y2 = self.compl_mul2d(x_ft[:, -self.modes1:, :self.modes2], complex_mult2)

        # y1 = self.compl_mul2d(x_ft[:, :self.modes1, :self.modes2], self.weights1)
        # y2 = self.compl_mul2d(x_ft[:, -self.modes1:, :self.modes2], self.weights2)

        out_ft = out_ft.at[:, :self.modes1, :self.modes2].set(y1)
        out_ft = out_ft.at[:, -self.modes1:, :self.modes2].set(y2)

        #Return to physical space
        x = jnp.fft.irfft2(out_ft, s=(shape[-2], shape[-1]))
        return x

class FNO_multimodal_2d(eqx.Module):
    dL : float
    omega : float
    modes1: int
    modes2: int
    width: int
    padding: int
    sizex: int
    sizey: int
    mod_data_channels: int
    pre_data_channels: int
    fc0_pre: eqx.nn.Linear
    num_fourier_layers: int
    ALPHA: float
    convs: list
    ws: list
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    m_basic1: eqx.Module
    m_basic2: eqx.Module
    m_basic3: eqx.Module
    m_pool1: eqx.nn.AvgPool2d
    m_pool2: eqx.nn.AvgPool2d
    m_pool3: eqx.nn.AvgPool2d
    m_bc1: eqx.nn.Linear
    m_bc2_1: eqx.nn.Linear
    m_bc2_2: eqx.nn.Linear
    def __init__(self, args, *, key):
        keys = jax.random.split(key, num=9+2*args.num_fourier_layers)

        self.dL = dL
        self.omega = omega
        
        self.modes1 = args.f_modes
        self.modes2 = args.f_modes
        self.width = args.HIDDEN_DIM
        self.padding = args.f_padding # pad the domain if input is non-periodic
        self.sizex = args.domain_sizex
        self.sizey = args.domain_sizey
        self.mod_data_channels = 6
        self.pre_data_channels = 6
        self.fc0_pre = eqx.nn.Linear(6, args.HIDDEN_DIM, key=keys[0]) # in_c, out_c
        self.num_fourier_layers = args.num_fourier_layers
        self.ALPHA = args.ALPHA

        self.m_basic1 = BasicBlock(self.mod_data_channels, self.width, self.ALPHA, 1, key=keys[1])
        self.m_basic2 = BasicBlock(self.width, self.width, self.ALPHA, 1, key=keys[2])
        self.m_basic3 = BasicBlock(self.width, self.width, self.ALPHA, 1, key=keys[3])
        self.m_pool1 = eqx.nn.AvgPool2d(2,2)
        self.m_pool2 = eqx.nn.AvgPool2d(2,2)
        self.m_pool3 = eqx.nn.AvgPool2d(2,2)
        self.m_bc1 = eqx.nn.Linear(int(self.width*self.sizex*self.sizey/64), self.modes1*self.modes2, key=keys[4])

        self.m_bc2_1 = eqx.nn.Linear(self.modes1*self.modes2, self.modes1*self.modes2, key=keys[5])
        self.m_bc2_2 = eqx.nn.Linear(self.modes1*self.modes2, self.modes1*self.modes2, key=keys[6])

        self.fc1 = eqx.nn.Linear(self.width, 128, key=keys[7])
        self.fc2 = eqx.nn.Linear(128, args.outc, key=keys[8])

        self.convs = []
        self.ws = []
        for i in range(self.num_fourier_layers):
            self.convs.append(Modulated_SpectralConv2d(self.width, self.width, self.modes1, self.modes2, key=keys[9+2*i]))
            # self.ws.append(eqx.nn.Conv2d(self.width, self.width, 3, padding=1, key=keys[10+2*i])) # spatial_dim, in_c, out_c, kernel_size
            self.ws.append(BasicBlock_without_shortcut(self.width, self.width, self.ALPHA, 1, key=keys[10+2*i]))

    @eqx.filter_jit
    def __call__(self, yeex, yeey, top_bc, bottom_bc, left_bc, right_bc):
        # Sx_f: [subdomain_size, subdomain_size,2]
        # Sy_f: [subdomain_size, subdomain_size,2]
        # source: [subdomain_size, subdomain_size, 2]
        # yeex: [subdomain_size, subdomain_size-2]
        # yeey: [subdomain_size-2, subdomain_size]
        # top_bc, bottom_bc: [1, subdomain_size, 2]
        # left_bc, right_bc: [subdomain_size, 1, 2]

        #pad 0s to yeex and yeey:
        # yeex = jnp.pad(yeex0, ((0,0),(1,1)), mode='constant', constant_values = 0)
        # yeey = jnp.pad(yeey0, ((1,1),(0,0)), mode='constant', constant_values = 0)

        preprocessed = jnp.zeros((2, yeex.shape[0], yeex.shape[1]))

        # just use bc itself and leave 0s in the middle
        for channel in [0,1]:
            preprocessed = preprocessed.at[channel, 0,1:-1].set(jnp.squeeze(top_bc,axis=0)[1:-1,channel])
            preprocessed = preprocessed.at[channel, -1,1:-1].set(jnp.squeeze(bottom_bc,axis=0)[1:-1,channel])
            preprocessed = preprocessed.at[channel, 1:-1,0].set(jnp.squeeze(left_bc,axis=1)[1:-1,channel])
            preprocessed = preprocessed.at[channel, 1:-1,-1].set(jnp.squeeze(right_bc,axis=1)[1:-1,channel])

        # print("shapes4: ", yeex.shape, preprocessed.shape)

        grid = self.get_grid(yeex.shape)
        # print("sh:" , source.shape, yeex.shape, yeey.shape, grid.shape, preprocessed.shape)
        # Sx_f = 1/Sx_f
        # Sy_f = 1/Sy_f

        pre_data = jnp.concatenate((jnp.expand_dims(yeex,axis=0), jnp.expand_dims(yeey, axis=0), grid, preprocessed), axis=0)
        mod_data = pre_data
        # modulating branch:
        mod = self.m_pool1(self.m_basic1(mod_data))
        mod = self.m_pool2(self.m_basic2(mod))
        mod = self.m_pool3(self.m_basic3(mod))
        mod = self.m_bc1(jnp.reshape(mod, (-1)))
        # mod1 = self.m_bc2_1(mod).reshape((batch_size, self.width,self.width,1,1))
        # mod2 = self.m_bc2_2(mod).reshape((batch_size, self.width,self.width,1,1))
        mod1 = self.m_bc2_1(mod).reshape((1,1, self.modes1,self.modes2))
        mod2 = self.m_bc2_2(mod).reshape((1,1, self.modes1,self.modes2))

        x = jax.vmap(jax.vmap(self.fc0_pre, 1, 1),2,2)(pre_data)
        
        if self.padding > 0:
            x = jnp.pad(x,  ((0,0), (0,self.padding), (0,self.padding)), mode='constant', constant_values = 0)

        for i in range(self.num_fourier_layers-1):
            x1 = self.convs[i](x.shape, mod1, mod2, x)
            x2 = self.ws[i](x)
            x = x1 + x2 + x
            x = jax.nn.leaky_relu(x, negative_slope=self.ALPHA)
            
        x1 = self.convs[-1](x.shape, mod1, mod2, x)
        x2 = self.ws[-1](x)
        x = x1 + x2 + x

        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]

        # x = x.transpose((1, 2, 0))
        x = jax.vmap(jax.vmap(self.fc1, 1, 1),2,2)(x)
        # x = self.fc1(x)
        x = jax.nn.leaky_relu(x, negative_slope=self.ALPHA)
        # x = jax.nn.gelu(x)
        x = jax.vmap(jax.vmap(self.fc2, 1, 1),2,2)(x)
        # x = self.fc2(x)
        x = x.transpose((1, 2, 0))

        return x

    @eqx.filter_jit
    def eval_forward(self, yeex, yeey, top_bc, bottom_bc, left_bc, right_bc):
        # Sx_f: [subdomain_size, subdomain_size,2]
        # Sy_f: [subdomain_size, subdomain_size,2]
        # source: [subdomain_size, subdomain_size, 2]
        # yeex: [subdomain_size, subdomain_size-2]
        # yeey: [subdomain_size-2, subdomain_size]
        # top_bc, bottom_bc: [1, subdomain_size, 2]
        # left_bc, right_bc: [subdomain_size, 1, 2]

        #pad 0s to yeex and yeey:
        # yeex = jnp.pad(yeex0, ((0,0),(1,1)), mode='constant', constant_values = 0)
        # yeey = jnp.pad(yeey0, ((1,1),(0,0)), mode='constant', constant_values = 0)

        preprocessed = jnp.zeros((2, yeex.shape[0], yeex.shape[1]))

        # just use bc itself and leave 0s in the middle
        for channel in [0,1]:
            preprocessed = preprocessed.at[channel, 0,1:-1].set(jnp.squeeze(top_bc,axis=0)[1:-1,channel])
            preprocessed = preprocessed.at[channel, -1,1:-1].set(jnp.squeeze(bottom_bc,axis=0)[1:-1,channel])
            preprocessed = preprocessed.at[channel, 1:-1,0].set(jnp.squeeze(left_bc,axis=1)[1:-1,channel])
            preprocessed = preprocessed.at[channel, 1:-1,-1].set(jnp.squeeze(right_bc,axis=1)[1:-1,channel])

        # print("shapes4: ", yeex.shape, preprocessed.shape)

        grid = self.get_grid(yeex.shape)
        # print("sh:" , source.shape, yeex.shape, yeey.shape, grid.shape, preprocessed.shape)
        # Sx_f = 1/Sx_f
        # Sy_f = 1/Sy_f
        
        pre_data = jnp.concatenate((jnp.expand_dims(yeex,axis=0), jnp.expand_dims(yeey, axis=0), grid, preprocessed), axis=0)
        mod_data = pre_data
        # modulating branch:
        mod = self.m_pool1(self.m_basic1(mod_data))
        mod = self.m_pool2(self.m_basic2(mod))
        mod = self.m_pool3(self.m_basic3(mod))
        mod = self.m_bc1(jnp.reshape(mod, (-1)))
        # mod1 = self.m_bc2_1(mod).reshape((batch_size, self.width,self.width,1,1))
        # mod2 = self.m_bc2_2(mod).reshape((batch_size, self.width,self.width,1,1))
        mod1 = self.m_bc2_1(mod).reshape((1,1, self.modes1,self.modes2))
        mod2 = self.m_bc2_2(mod).reshape((1,1, self.modes1,self.modes2))

        x = jax.vmap(jax.vmap(self.fc0_pre, 1, 1),2,2)(pre_data)
        
        if self.padding > 0:
            x = jnp.pad(x,  ((0,0), (0,self.padding), (0,self.padding)), mode='constant', constant_values = 0)

        for i in range(self.num_fourier_layers-1):
            x1 = self.convs[i](x.shape, mod1, mod2, x)
            x2 = self.ws[i](x)
            x = x1 + x2 + x
            x = jax.nn.leaky_relu(x, negative_slope=self.ALPHA)
            
        x1 = self.convs[-1](x.shape, mod1, mod2, x)
        x2 = self.ws[-1](x)
        x = x1 + x2 + x

        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]

        # x = x.transpose((1, 2, 0))
        x = jax.vmap(jax.vmap(self.fc1, 1, 1),2,2)(x)
        # x = self.fc1(x)
        x = jax.nn.leaky_relu(x, negative_slope=self.ALPHA)
        # x = jax.nn.gelu(x)
        x = jax.vmap(jax.vmap(self.fc2, 1, 1),2,2)(x)
        # x = self.fc2(x)
        x = x.transpose((1, 2, 0))

        return x, preprocessed, grid

    def get_grid(self, shape):
        size_x, size_y = shape[0], shape[1]
        gridx = jnp.tile(jnp.expand_dims(jnp.linspace(0, 1, size_x),1), (1,size_y))
        # gridx = gridx.reshape(1, size_x, 1).repeat(jnp.asarray([1, 1, size_y]))
        gridy = jnp.tile(jnp.expand_dims(jnp.linspace(0, 1, size_y),0), (size_x,1))
        # gridy = gridy.reshape(1, 1, size_y).repeat(jnp.asarray([1, size_x, 1]))
        return jnp.stack((gridx, gridy), axis=0)
