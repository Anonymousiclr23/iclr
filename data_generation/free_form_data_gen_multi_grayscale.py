import numpy as np
import random
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import ceviche
from ceviche import fdfd_hz
from ceviche.constants import C_0
import time
import sys,os

# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["KMP_WARNINGS"] = "FALSE" 

# Define some constants
wl = 1050
wavelength = wl*1e-9
omega = 2 * np.pi * C_0 / wavelength
dL = np.array([6.25, 6.25])

Nx = int(sys.argv[1])
Ny = int(sys.argv[1])
grid_shape = Nx, Ny

pml_x = 40
pml_y = pml_x

space_x = int(float(sys.argv[2])*wl/dL[0])
space_y = int(float(sys.argv[2])*wl/dL[1])

device_pxls_x = Nx-2*pml_x-2*space_x
device_pxls_y = Ny-2*pml_y-2*space_y

assert device_pxls_x<2000
assert device_pxls_y<2000

npml = [pml_x, pml_y] # Periodic in x direction
eps_r = np.ones(grid_shape)

eps_Si = 12.726
eps_Ge = 16
eps_air = 1

def greyscale_paint_1(pattern, max_region=None,seed=0):
    # use several different strategy to paint the binary patterns into greyscale patterns
    # Strat (1):
    # use random Fourier waves in 2d to generate continuous map
    np.random.seed(seed)

    from voronoi import generate

    for idx,p in enumerate(pattern):
        print("p.shape: ", p.shape)
        min_region = max(1,int(p.shape[1]/500)**2)
        max_region = max(2,int(p.shape[1]/30)**2) if not max_region else max_region
        
        N_regions = np.random.randint(min_region,max_region)
        voronoi_map = generate(
                width = p.shape[1],
                height = p.shape[2],
                regions = N_regions,
                colors = np.random.rand(N_regions)*(eps_Ge-eps_air)
            )
        pattern[idx] = voronoi_map*p[0]+eps_air

    return pattern

def greyscale_paint_2(pattern, alpha, seed=0):
    np.random.seed(seed)
    from gaussian_random_fields import gaussian_random_field
    # use several different strategy to paint the binary patterns into greyscale patterns
    # Strat (2):
    # use random Gaussian fields

    for idx,p in enumerate(pattern):
        GRM = gaussian_random_field(alpha=alpha, size=p.shape[1])
        GRM = GRM-np.min(GRM)
        GRM = GRM/np.max(GRM)
        GRM = GRM*(eps_Ge-eps_air)

        pattern[idx] = GRM*p[0]+eps_air

    return pattern


def data_generation(full_pattern: np.array) -> np.array:
    if device_pxls_x>0 and device_pxls_y>0:
        eps_r[space_x+pml_x:space_x+pml_x+device_pxls_x, space_y+pml_y:space_y+pml_y+device_pxls_y] = full_pattern[:, :device_pxls_x, :device_pxls_y]
    
    # Set up the FDFD simulation for TM
    F = fdfd_hz(omega, dL[0]*1e-9, eps_r, npml)
    
    # Source
    source_amp = 64e9/dL[0]/dL[1]
    random_source = np.zeros(grid_shape, dtype=complex)
    k0 = 2 * np.pi / wavelength
    n_angles = 4 # number of different angles in each line source
    

    src_space = 10
    src_offsetx = pml_x + int(1/2*space_x)
    src_offsety = pml_y + int(1/2*space_y)

    #top source
    source_amp_y = np.zeros(Ny-2*src_offsety-2*src_space, dtype=complex)
    for j in range(n_angles):
        angle_deg = random.randint(-90,90)

        angle_rad = angle_deg * np.pi / 180
        # Compute the wave vector
        kx = k0 * np.cos(angle_rad)
        ky = k0 * np.sin(angle_rad)

        # Get an array of the y positions across the simulation domain
        Ly = (Ny-2*src_offsety-2*src_space) * dL[0]*1e-9
        y_vec = np.linspace(0, Ly, Ny-2*src_offsety-2*src_space)
        
        # Make a new source where source[x] ~ exp(i * kx * x) to simulate an angle
        phase = 2*np.pi*random.random()
        source_amp_y += np.exp(1j * ky * y_vec + 1j*phase)
    
    source_amp_y *= 1/n_angles
    random_source[src_offsetx, src_offsety+src_space:Ny-src_offsety-src_space] = source_amp * source_amp_y

    #bottom source
    source_amp_y = np.zeros(Ny-2*src_offsety-2*src_space, dtype=complex)
    for j in range(n_angles):
        angle_deg = random.randint(-90,90)

        angle_rad = angle_deg * np.pi / 180
        # Compute the wave vector
        kx = k0 * np.cos(angle_rad)
        ky = k0 * np.sin(angle_rad)

        # Get an array of the y positions across the simulation domain
        Ly = (Ny-2*src_offsety-2*src_space) * dL[0]*1e-9
        y_vec = np.linspace(0, Ly, Ny-2*src_offsety-2*src_space)
        
        # Make a new source where source[x] ~ exp(i * kx * x) to simulate an angle
        phase = 2*np.pi*random.random()
        source_amp_y += np.exp(1j * ky * y_vec + 1j*phase)
    
    source_amp_y *= 1/n_angles
    random_source[Nx-1-src_offsetx, src_offsety+src_space:Ny-src_offsety-src_space] = source_amp * source_amp_y

    #left source
    source_amp_x = np.zeros(Nx-2*src_offsetx-2*src_space, dtype=complex)
    for j in range(n_angles):
        angle_deg = random.randint(-90,90)

        angle_rad = angle_deg * np.pi / 180
        # Compute the wave vector
        kx = k0 * np.cos(angle_rad)
        ky = k0 * np.sin(angle_rad)

        # Get an array of the y positions across the simulation domain
        Lx = (Nx-2*src_offsetx-2*src_space) * dL[0]*1e-9
        x_vec = np.linspace(0, Lx, Nx-2*src_offsetx-2*src_space)
        
        # Make a new source where source[x] ~ exp(i * kx * x) to simulate an angle
        phase = 2*np.pi*random.random()
        source_amp_x += np.exp(1j * kx * x_vec + 1j*phase)
    
    source_amp_x *= 1/n_angles
    random_source[src_offsetx+src_space:Nx-src_offsetx-src_space, src_offsety] = source_amp * source_amp_x

    #right source
    source_amp_x = np.zeros(Nx-2*src_offsetx-2*src_space, dtype=complex)
    for j in range(n_angles):
        angle_deg = random.randint(-90,90)

        angle_rad = angle_deg * np.pi / 180
        # Compute the wave vector
        kx = k0 * np.cos(angle_rad)
        ky = k0 * np.sin(angle_rad)

        # Get an array of the y positions across the simulation domain
        Lx = (Nx-2*src_offsetx-2*src_space) * dL[0]*1e-9
        x_vec = np.linspace(0, Lx, Nx-2*src_offsetx-2*src_space)
        
        # Make a new source where source[x] ~ exp(i * kx * x) to simulate an angle
        phase = 2*np.pi*random.random()
        source_amp_x += np.exp(1j * kx * x_vec + 1j*phase)

    source_amp_x *= 1/n_angles
    random_source[src_offsetx+src_space:Nx-src_offsetx-src_space, Ny-src_offsety] = source_amp * source_amp_x

    # Solve the FDFD simulation for the fields, offset the phase such that Ex has 0 phase at the center of the bottom row of the window
    Ex_forward, Ey_forward, Hz_forward = F.solve(random_source)

    return eps_r, Hz_forward, Ex_forward, Ey_forward, random_source

if __name__ == '__main__':

    input_dir = "free_form_shapes"
    output_dir = "large_scale_data/"+str(sys.argv[1])+"_"+str(sys.argv[1])
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    full_pattern = np.load(input_dir+"/"+"freeform_gen_2000_2000_binary_tilted_50devices.npy")

    grayscale_pattern = np.concatenate((greyscale_paint_1(full_pattern[:,:,:device_pxls_x,:device_pxls_y].copy(), max_region=10, seed=0),
                                        greyscale_paint_1(full_pattern[:,:,:device_pxls_x,:device_pxls_y].copy(), max_region=10, seed=1),
                                        greyscale_paint_1(full_pattern[:,:,:device_pxls_x,:device_pxls_y].copy(), max_region=20, seed=2),
                                        greyscale_paint_1(full_pattern[:,:,:device_pxls_x,:device_pxls_y].copy(), max_region=20, seed=3),
                                        greyscale_paint_1(full_pattern[:,:,:device_pxls_x,:device_pxls_y].copy(), max_region=50, seed=4),
                                        greyscale_paint_1(full_pattern[:,:,:device_pxls_x,:device_pxls_y].copy(), max_region=50, seed=5),
                                        greyscale_paint_1(full_pattern[:,:,:device_pxls_x,:device_pxls_y].copy(), max_region=100, seed=6),
                                        greyscale_paint_1(full_pattern[:,:,:device_pxls_x,:device_pxls_y].copy(), max_region=100, seed=7),
                                        greyscale_paint_1(full_pattern[:,:,:device_pxls_x,:device_pxls_y].copy(), max_region=400, seed=8),
                                        greyscale_paint_1(full_pattern[:,:,:device_pxls_x,:device_pxls_y].copy(), max_region=400, seed=9),
                                        greyscale_paint_2(full_pattern[:,:,:device_pxls_x,:device_pxls_y].copy(), alpha=4, seed=10),
                                        greyscale_paint_2(full_pattern[:,:,:device_pxls_x,:device_pxls_y].copy(), alpha=4, seed=11),
                                        greyscale_paint_2(full_pattern[:,:,:device_pxls_x,:device_pxls_y].copy(), alpha=4.5, seed=12),
                                        greyscale_paint_2(full_pattern[:,:,:device_pxls_x,:device_pxls_y].copy(), alpha=4.5, seed=13),
                                        greyscale_paint_2(full_pattern[:,:,:device_pxls_x,:device_pxls_y].copy(), alpha=5, seed=14),
                                        greyscale_paint_2(full_pattern[:,:,:device_pxls_x,:device_pxls_y].copy(), alpha=5, seed=15),
                                        greyscale_paint_2(full_pattern[:,:,:device_pxls_x,:device_pxls_y].copy(), alpha=5.5, seed=16),
                                        greyscale_paint_2(full_pattern[:,:,:device_pxls_x,:device_pxls_y].copy(), alpha=5.5, seed=17),
                                        greyscale_paint_2(full_pattern[:,:,:device_pxls_x,:device_pxls_y].copy(), alpha=6, seed=18),
                                        greyscale_paint_2(full_pattern[:,:,:device_pxls_x,:device_pxls_y].copy(), alpha=6, seed=19)
                                        ), axis=0)

    total_N = grayscale_pattern.shape[0]

    tic = time.time()

    each_device_time =[]
    # Initialize output fields
    input_eps = np.empty([total_N, Nx, Ny], dtype = np.float32)
    Hz_out_forward = np.empty([total_N, Nx, Ny], dtype = complex)
    Ex_out_forward = np.empty([total_N, Nx, Ny], dtype = complex)
    Ey_out_forward = np.empty([total_N, Nx, Ny], dtype = complex)
    source = np.empty([total_N, Nx, Ny], dtype = complex)

    for i in range(total_N):
        print(i)
        tic_i = time.time()
        input_eps[i], Hz_out_forward[i], Ex_out_forward[i], Ey_out_forward[i], source[i] = data_generation(grayscale_pattern[i])
        toc_i = time.time()
        each_device_time.append(toc_i - tic_i)

    Hz_out_forward_RI = np.stack((np.real(Hz_out_forward), np.imag(Hz_out_forward)), axis = -1)
    Ex_out_forward_RI = np.stack((np.real(Ex_out_forward), np.imag(Ex_out_forward)), axis = -1)
    Ey_out_forward_RI = np.stack((np.real(Ey_out_forward), np.imag(Ey_out_forward)), axis = -1)
    source_RI = np.stack((np.real(source), np.imag(source)), axis = -1)
    
    toc = time.time()
    print(f"Device finished: {total_N}, The total time of the data generation is {toc - tic}s")

    each_device_time = np.array(each_device_time)
    print(f"for simulation domain of size {str(sys.argv[1])} by {str(sys.argv[1])}, Si device size of {str(device_pxls_x)} by {str(device_pxls_x)} (or 0 by 0 if negative), average time is {np.mean(each_device_time)}, variance is {np.var(each_device_time)}")

    pml_map = np.zeros((Nx, Ny), dtype=np.float32)
    pml_map[:pml_x+1,:] = 1
    pml_map[-pml_x+1:,:] = 1
    pml_map[:,:pml_y+1] = 1
    pml_map[:,-pml_y+1:] = 1

    np.save(output_dir+"/"+f"input_eps.npy", input_eps)
    np.save(output_dir+"/"+f"Hz_out_forward_RI.npy", Hz_out_forward_RI)
    np.save(output_dir+"/"+f"Ex_out_forward_RI.npy", Ex_out_forward_RI)
    np.save(output_dir+"/"+f"Ey_out_forward_RI.npy", Ey_out_forward_RI)
    np.save(output_dir+"/"+f"source_RI.npy", source_RI)
    np.save(output_dir+"/"+f"pml_map.npy", pml_map)

