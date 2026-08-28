# ========================================================================
#           On the Missing Area of the Bullet Cluster
#                         Brahim Benaissa
#                   https://Justpeers.com
#
#           Based on the Baryonic model of Famaey 2026
#                       arXiv:2605.10022
# ========================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from scipy.io import savemat
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.special import erf
import warnings

# Suppress runtime warnings for division by zero or invalid values handled by code
np.seterr(divide='ignore', invalid='ignore')

# ========================================================================
#  Section 1: CONFIGURATION
# ========================================================================

# --- Mezzi Parameters ---
Compliance            = 377               # Mezzi's Compliance Constant

# --- Lensing Smoothing ---
Sigma_smoothing       = False     # set "True" to apply Lensing Smoothing
smoothing_fwhm_arcsec = 10        # 0 = No smoothing

# --- Fundamental Units ---
G     = 6.67430e-11
c     = 2.99792458e8
pc    = 3.085677581491367e16
kpc   = 1e3 * pc
Mpc   = 1e6 * pc
Msun  = 1.98847e30
arcsec_to_kpc = 4.413       # Rihtaršič 2026

# --- Simulation Grid & Box Parameters ---
Box_3d                = 40 * Mpc           # 3D box size in meters
N_2d                  = 5000               #  2D adaptive grid resolution
N_z                   = 2000               #  Line of sight grid resolution

# --- Dual-focus adaptive grid using inverse CDF ---
stretch_factor        = 5                 # Adaptive grid stretching factor, 0 = No Strech
focus_width           = 200e3 * pc        # High-res focus width for 2D grid (x,y)
focus_width_z         = 40e3 * pc         # High-res focus width for z-axis

# --- FFT / Uniform Proxy Grid Parameters ---
N_fft                 = N_2d              # Match primary grid dimensions
fft_box_half_size_kpc = 4000              # Half-size of the uniform FFT box in kpc

# --- Ray Tracing Parameters ---
N_theta               = 30                # Number of polar angles for rays
N_phi                 = 60                # Number of azimuthal angles for rays
N_r_ray               = 300               # Number of radial steps along each ray

print('\n================================================================================')
print('   On the Missing Area of the Bullet Cluster')
print('   [Famaey 2026 Mass Model]')
print('================================================================================\n')

# ========================================================================
#  LOCAL FUNCTIONS
# ========================================================================

def trace_component_K(comp, Phi, M_input, X_3D, Y_3D, Z_3D, r_ray, R_box, c_speed, Compliance, N_theta, N_phi):
    G = 6.67430e-11
    xc, yc, zc = comp['x_c'], comp['y_c'], comp['z_c']
    Xs = X_3D - xc; Ys = Y_3D - yc; Zs = Z_3D - zc
    R_sph = np.sqrt(Xs**2 + Ys**2 + Zs**2)

    R_flat = R_sph.flatten()
    X_flat = Xs.flatten()
    Y_flat = Ys.flatten()
    Z_flat = Zs.flatten()

    theta_bin = np.linspace(0, np.pi, N_theta+1)
    phi_bin = np.linspace(0, 2*np.pi, N_phi+1)

    theta_pts = np.arccos(np.clip(Z_flat / np.maximum(R_flat, 1e-10), -1, 1))
    phi_pts = np.mod(np.arctan2(Y_flat, X_flat), 2*np.pi)

    t_idx = np.digitize(theta_pts, theta_bin)
    p_idx = np.digitize(phi_pts, phi_bin)

    # MATLAB sub2ind uses 1-based indexing. Python is 0-based.
    dir_idx = (t_idx - 1) * N_phi + (p_idx - 1)
    dir_idx[(t_idx == 0) | (t_idx > N_theta) | (p_idx == 0) | (p_idx > N_phi)] = 0

    N_dirs = N_theta * N_phi

    # accumarray replacement
    n_x = np.zeros(N_dirs)
    n_y = np.zeros(N_dirs)
    n_z = np.zeros(N_dirs)
    np.add.at(n_x, dir_idx, X_flat)
    np.add.at(n_y, dir_idx, Y_flat)
    np.add.at(n_z, dir_idx, Z_flat)
    counts = np.bincount(dir_idx, minlength=N_dirs)

    valid = counts > 0
    n_x[valid] /= counts[valid]
    n_y[valid] /= counts[valid]
    n_z[valid] /= counts[valid]

    norm_n = np.sqrt(n_x**2 + n_y**2 + n_z**2)
    empty_bins = norm_n == 0
    n_x[empty_bins] = 1; n_y[empty_bins] = 0; n_z[empty_bins] = 0; norm_n[empty_bins] = 1

    n_x /= norm_n; n_y /= norm_n; n_z /= norm_n

    # Shape: (N_r_ray, N_dirs)
    x_ray = r_ray[:, np.newaxis] * n_x
    y_ray = r_ray[:, np.newaxis] * n_y
    z_ray = r_ray[:, np.newaxis] * n_z

    # Add center and evaluate Phi
    # Phi expects shape (N_r, N_dirs, 1)
    Phi_ray = Phi(x_ray[:,:,np.newaxis] + xc, y_ray[:,:,np.newaxis] + yc, z_ray[:,:,np.newaxis] + zc)

    outside = r_ray > R_box
    tail_vals = -G * M_input / r_ray[outside]
    Phi_ray[outside, :] = tail_vals[:, np.newaxis]

    safe_Phi = np.maximum(np.abs(Phi_ray), 1e-10)
    integrand = (Phi_ray / safe_Phi) * np.sqrt(2 * np.abs(Phi_ray)) / (c_speed * r_ray[:, np.newaxis])

    cum_int = cumulative_trapezoid(integrand, r_ray, axis=0, initial=0)
    total_int = cum_int[-1, :]

    # Interpolation for cum_at_R
    idx_lo = np.searchsorted(r_ray, R_flat, side='right') - 1
    idx_lo = np.clip(idx_lo, 0, len(r_ray)-2)
    idx_hi = idx_lo + 1

    r_lo = r_ray[idx_lo]
    r_hi = r_ray[idx_hi]
    denom = r_hi - r_lo
    denom[denom == 0] = 1
    w = (R_flat - r_lo) / denom

    cum_at_R = cum_int[idx_lo, dir_idx] + w * (cum_int[idx_hi, dir_idx] - cum_int[idx_lo, dir_idx])

    overshoot = R_flat >= r_ray[-1]
    if np.any(overshoot):
        cum_at_R[overshoot] = cum_int[-1, dir_idx[overshoot]]

    CumP = total_int[dir_idx] - cum_at_R
    CumP[R_flat > r_ray[-1]] = 0
    K_2D = np.exp(CumP)

    K_3D = K_2D.reshape(X_3D.shape)
    Zeta_3D = np.exp(Compliance * (K_3D - 1))

    return K_3D, Zeta_3D


def generate_dual_focus_faces(u_faces, Box_half, x1, x2, sigma):
    L = Box_half
    baseline = 0.05
    u_norm = (u_faces + 1) / 2
    x_dense = np.linspace(-L, L, 500000)

    term1 = (sigma * np.sqrt(np.pi)/2) * (erf((x_dense - x1)/sigma) - erf((-L - x1)/sigma))
    term2 = (sigma * np.sqrt(np.pi)/2) * (erf((x_dense - x2)/sigma) - erf((-L - x2)/sigma))
    term3 = baseline * (x_dense - (-L))

    cdf = term1 + term2 + term3
    cdf = cdf / cdf[-1]

    x_faces = np.interp(u_norm, cdf, x_dense)
    return x_faces


def sampleflattened_famaey(N, a, x0, y0, seed, qxyz, PAdeg):
    np.random.seed(seed)
    u = np.random.rand(N)
    r = a / np.sqrt(u**(-2.0/3.0) - 1.0)
    costh = np.random.rand(N) * 2.0 - 1.0
    sinth = np.sqrt(1.0 - costh**2)
    phi = np.random.rand(N) * 2.0 * np.pi
    nx = sinth * np.cos(phi)
    ny = sinth * np.sin(phi)
    nz = costh

    qx, qy, qz = qxyz
    sx = qx * nx; sy = qy * ny; sz = qz * nz
    norm = np.sqrt(sx**2 + sy**2 + sz**2)
    nx2 = sx / norm; ny2 = sy / norm; nz2 = sz / norm

    pa = np.radians(PAdeg)
    cp, sp = np.cos(pa), np.sin(pa)
    rx = cp * nx2 - sp * ny2
    ry = sp * nx2 + cp * ny2
    rz = nz2

    x = x0 + r * rx
    y = y0 + r * ry
    z = r * rz
    return x, y, z, r


def snaptotargets_famaey(x, y, z, rorig, targets, cx, cy):
    N = len(x)
    available = np.ones(N, dtype=bool)
    for t in range(targets.shape[0]):
        xt, yt = targets[t, 0], targets[t, 1]
        if not np.any(available): break
        idxs = np.where(available)[0]
        d2 = (x[idxs] - xt)**2 + (y[idxs] - yt)**2
        min_idx = np.argmin(d2)
        j = idxs[min_idx]
        available[j] = False
        x[j] = xt; y[j] = yt
        rho2 = (xt - cx)**2 + (yt - cy)**2
        z2 = rorig[j]**2 - rho2
        if z2 >= 0.0:
            z[j] = np.sqrt(z2) if z[j] >= 0.0 else -np.sqrt(z2)
        else:
            z[j] = 0.0
    return x, y, z


# ========================================================================
#  Section 2: BULLET CLUSTER MASS MODEL
# ========================================================================

# 2.1: COSMOLOGY
H0 = 70.0; Omega_m = 0.30; Omega_L = 0.70
def E_z(z): return np.sqrt(Omega_m * (1+z)**3 + Omega_L)

# Cosmology (Famaey 2026 places all sources at infinity)
z_lens = 0.296

# Angular diameter distance to the lens
E_z_lens = np.sqrt(Omega_m * (1+z_lens)**3 + Omega_L)
z_vec = np.linspace(0, z_lens, 2000)
D_C_lens = (c/1000)/H0 * trapezoid(1.0/np.sqrt(Omega_m * (1+z_vec)**3 + Omega_L), z_vec)
D_L  = D_C_lens / (1+z_lens) * Mpc

# Critical surface density (match Famaey 2026)
Sigma_crit = 1827   # 1.827e9 Msun / kpc^2 = 1827 Msun / pc^2

# 2.2: LOAD CATALOG
cat_filename = 'cluster_members_specz_cat.dat'

if not os.path.isfile(cat_filename):
    raise FileNotFoundError(f'File not found: {cat_filename}\nObtain full catalog from Rihtaršič+2026 data release.')

print(f'[LOAD] Reading {cat_filename}...')
# Mimicking textscan
data = np.genfromtxt(cat_filename, comments='#', skip_header=1, dtype=None, encoding=None)
gal_id    = np.array([d[0] for d in data])
gal_ra    = np.array([d[1] for d in data])
gal_dec   = np.array([d[2] for d in data])
gal_mag   = np.array([d[3] for d in data])

N_cat = len(gal_id)
print(f'[LOAD] {N_cat} cluster members loaded.')

# 2.3: BCG IDENTIFICATION
sort_mag = np.argsort(gal_mag)

# BCG1: brightest galaxy, defines coordinate origin
bcg1_idx = sort_mag[0]
RA_BCG1  = gal_ra[bcg1_idx]
Dec_BCG1 = gal_dec[bcg1_idx]
cos_dec  = np.cos(np.radians(Dec_BCG1))

# Convert ALL catalog positions to Famaey frame: x=west, y=north, in kpc
x_cat = -(gal_ra - RA_BCG1) * 3600 * cos_dec * arcsec_to_kpc
y_cat = (gal_dec - Dec_BCG1) * 3600 * arcsec_to_kpc

# BCG3: brightest galaxy near expected (820, 220) kpc
dist_bcg3_exp = np.sqrt((x_cat - 820)**2 + (y_cat - 220)**2)
dist_bcg3_exp[bcg1_idx] = np.inf
bcg3_idx = np.argmin(dist_bcg3_exp)

# BCG2: brightest galaxy near expected (106, 129) kpc
dist_bcg2_exp = np.sqrt((x_cat - 106)**2 + (y_cat - 129)**2)
dist_bcg2_exp[[bcg1_idx, bcg3_idx]] = np.inf
bcg2_idx = np.argmin(dist_bcg2_exp)

print('[BCG] Identified from catalog:')
print(f'      BCG1 (ID {gal_id[bcg1_idx]}): mag={gal_mag[bcg1_idx]:.2f}, pos=({x_cat[bcg1_idx]:7.1f}, {y_cat[bcg1_idx]:7.1f}) kpc')
print(f'      BCG2 (ID {gal_id[bcg2_idx]}): mag={gal_mag[bcg2_idx]:.2f}, pos=({x_cat[bcg2_idx]:7.1f}, {y_cat[bcg2_idx]:7.1f}) kpc [exp: ~106, 129]')
print(f'      BCG3 (ID {gal_id[bcg3_idx]}): mag={gal_mag[bcg3_idx]:.2f}, pos=({x_cat[bcg3_idx]:7.1f}, {y_cat[bcg3_idx]:7.1f}) kpc [exp: ~820, 220]')

# BCG properties (Famaey Section 2.3)
M_bcg  = 1.0e12 * Msun
r_bcg  = 10e3 * pc

# 2.4: MASS BUDGET
M_main_smooth = 1.196e13 * Msun   # main galaxy component (incl. BCG1, BCG2)
M_sub_smooth  = 4.0e12  * Msun    # subcluster galaxy component (incl. BCG3)

M_remain_main = M_main_smooth - 2*M_bcg   # BCG1 + BCG2
M_remain_sub  = M_sub_smooth  - 1*M_bcg   # BCG3

N_main_total = 166   # non-BCG main cluster galaxies
N_sub_total  = 50    # non-BCG subcluster galaxies

# Uniform mass per galaxy
gal_mass_uniform = M_remain_main / N_main_total  # = 6.0e10 Msun by construction

assert abs(gal_mass_uniform - M_remain_sub/N_sub_total) < 1e-6 * gal_mass_uniform, 'Mass budget inconsistent'

print(f'[MASS] Uniform galaxy mass: {gal_mass_uniform:.4e} Msun (= {gal_mass_uniform/Msun/1e10:.2f} x 10^10)')

# 2.5: PLUMMER PARAMETERS
a_main = 470e3 * pc   # smooth model Plummer radius, main component
a_sub  = 245e3 * pc   # smooth model Plummer radius, subcluster
gal_rs = 3e3 * pc     # all non-BCG galaxies

# 2.6: INITIALIZE OUTPUT DICT
components = {}

# --- GAS COMPONENTS (No-Taper: Flat Offset Profile) ---
components[1] = {'name':'Main Gas',         'x_c':190e3*pc, 'y_c':90e3*pc,  'z_c':0, 'type':'Plummer', 'M_total':2.0e14*Msun,  'r_s':565e3*pc}
components[2] = {'name':'Subcluster Gas 1', 'x_c':525e3*pc, 'y_c':120e3*pc, 'z_c':0, 'type':'Plummer', 'M_total':1.5e13*Msun,  'r_s':505e3*pc}
components[3] = {'name':'Subcluster Gas 2', 'x_c':620e3*pc, 'y_c':165e3*pc, 'z_c':0, 'type':'Plummer', 'M_total':4.1e12*Msun,  'r_s':100e3*pc}
components[4] = {'name':'Sub Gas 3',        'x_c':720e3*pc, 'y_c':175e3*pc, 'z_c':0, 'type':'Plummer', 'M_total':4.1e12*Msun,  'r_s':100e3*pc}

# --- BCGs ---
components[5] = {'name':'BCG 1', 'x_c':x_cat[bcg1_idx]*kpc, 'y_c':y_cat[bcg1_idx]*kpc, 'z_c':0, 'type':'Plummer', 'M_total':M_bcg, 'r_s':r_bcg}
components[6] = {'name':'BCG 2', 'x_c':x_cat[bcg2_idx]*kpc, 'y_c':y_cat[bcg2_idx]*kpc, 'z_c':0, 'type':'Plummer', 'M_total':M_bcg, 'r_s':r_bcg}
components[7] = {'name':'BCG 3', 'x_c':820*kpc, 'y_c':220*kpc, 'z_c':0, 'type':'Plummer', 'M_total':M_bcg, 'r_s':r_bcg}

# Subcluster centre for coordinate transforms (Hardcoded to match Famaey 2026)
x_sub_c_kpc = 820
y_sub_c_kpc = 220

# 2.7: MODE-SPECIFIC GALAXY GENERATION
print('\n[MODE] === Famaey No-Taper (Flat Offset Profile) ===')

cx_sub = x_sub_c_kpc * kpc
cy_sub = y_sub_c_kpc * kpc

SNAPTARGETSMAINXY = np.array([
    [0.0, 10.0], [5.0, 22.0], [65.0, 39.2], [78.5, 9.0],
    [56.6, 80.1], [102.3, 29.1], [62.6, 90.1], [43.2, 117.6],
    [96.3, 96.8], [64.2, 123.5], [76.4, 121.8], [5.7, -24.4],
    [-46.4, -25.3], [38.2, -41.2], [-59.5, -4.2], [59.8, -2.7],
    [68.4, -25.7], [15.9, -77.3], [-51.2, -50.3], [4.0, -88.1],
    [-19.9, -94.5], [9.7, -108.2], [-106.8, -21.7], [30.2, -115.6],
    [-119.2, -36.8], [-81.0, -96.4], [107.6, 67.3], [-25.3, -126.1],
    [-9.7, 130.0], [-136.6, 24.0], [94.3, -107.6], [142.2, 33.6],
    [-106.0, 111.4], [-76.2, -140.8], [147.5, -70.2], [-124.8, 118.7],
    [170.6, -27.9], [33.5, 170.4], [98.0, 144.0], [165.0, 64.0],
    [141.0, 121.0], [117.9, 149.5], [187.2, 61.0], [179.2, 84.6],
    [150.4, -135.5], [178.6, 103.1], [192.0, 82.0], [152.0, 161.9],
    [87.6, 213.8], [20.6, 232.4], [41.1, 242.9], [-4.0, 261.8],
    [237.9, 128.3], [111.6, 254.1], [187.7, 223.4], [254.9, 151.9],
    [229.5, 213.8], [255.9, 268.2], [-82.4, 181.3], [25.0, 347.4],
    [161.3, 328.2]
])

SNAPTARGETSSUBXY = np.array([
    [820.0, 210.0], [770.3, 211.6], [846.0, 256.4], [777.9, 144.1],
    [864.3, 277.2], [863.5, 161.4], [738.5, 239.4], [904.2, 241.2],
    [731.0, 222.9], [910.5, 218.5], [750.0, 293.6], [894.3, 149.6],
    [804.2, 313.8], [851.7, 321.5], [739.9, 304.5], [755.2, 335.8],
    [696.2, 338.5], [648.3, 229.7], [653.7, 296.7], [631.9, 244.7],
    [688.8, 407.7], [510.2, 248.4]
])

N_t_main = SNAPTARGETSMAINXY.shape[0]
N_t_sub  = SNAPTARGETSSUBXY.shape[0]
print(f'       Targets: {N_t_main} main + {N_t_sub} sub = {N_t_main+N_t_sub} total')

# Generate main cluster galaxies
qxyz_main = [1.0, 0.35, 1.0]; PA_main = 50.0
xs, ys, zs, r0 = sampleflattened_famaey(N_main_total, a_main, 0.0, 0.0, 42, qxyz_main, PA_main)
xs, ys, zs = snaptotargets_famaey(xs, ys, zs, r0, SNAPTARGETSMAINXY * kpc, 0.0, 0.0)

# Generate subcluster galaxies
qxyz_sub = [1.0, 0.35, 1.0]; PA_sub = 0.0
xs2, ys2, zs2, r02 = sampleflattened_famaey(N_sub_total, a_sub, cx_sub, cy_sub, 43, qxyz_sub, PA_sub)
xs2, ys2, zs2 = snaptotargets_famaey(xs2, ys2, zs2, r02, SNAPTARGETSSUBXY * kpc, cx_sub, cy_sub)

# Assemble with uniform mass
comp_idx = 7
for i in range(N_main_total):
    comp_idx += 1
    components[comp_idx] = {'name':f'Main Gal {i+1}', 'x_c':xs[i], 'y_c':ys[i], 'z_c':zs[i], 'type':'Plummer', 'M_total':gal_mass_uniform, 'r_s':gal_rs}
for i in range(N_sub_total):
    comp_idx += 1
    components[comp_idx] = {'name':f'Sub Gal {i+1}', 'x_c':xs2[i], 'y_c':ys2[i], 'z_c':zs2[i], 'type':'Plummer', 'M_total':gal_mass_uniform, 'r_s':gal_rs}

# 2.8: VERIFICATION & POST-PROCESSING
N_comp = len(components)
print(f'\n[VERIFY] Total components: {N_comp}')
print(f'         Gas: 4 | BCGs: 3 | Galaxies: {N_comp - 7}')

M_total_check = sum(components[i]['M_total'] for i in range(5, N_comp+1))
M_expected = M_main_smooth + M_sub_smooth

assert abs(M_total_check - M_expected) < 1e-9 * M_expected, 'MASS CONSERVATION FAILED'
print(f'[VERIFY] Mass conservation verified (diff={abs(M_total_check-M_expected)/M_expected:.2e})')

star_idx = [k for k, v in components.items() if v['type']=='Plummer' and ('Gal' in v['name'] or 'BCG' in v['name'])]
gas_idx  = [k for k, v in components.items() if v['type']=='Plummer' and 'Gas' in v['name']]

comp_xy_kpc = np.zeros((N_comp, 2))
for j in range(1, N_comp+1):
    comp_xy_kpc[j-1, 0] = components[j]['x_c'] / kpc
    comp_xy_kpc[j-1, 1] = components[j]['y_c'] / kpc

names_comp = [components[j]['name'] for j in range(1, N_comp+1)]
idx_bcg1 = names_comp.index('BCG 1') + 1
idx_bcg2 = names_comp.index('BCG 2') + 1
idx_bcg3 = names_comp.index('BCG 3') + 1

# 2.9: BRIDGE: MAP COMPONENTS TO PIPELINE MACRO-VARIABLES
print('\n[BRIDGE] Mapping discrete components to pipeline variables...')

idx_main_gas = names_comp.index('Main Gas') + 1
idx_sub_gas1 = names_comp.index('Subcluster Gas 1') + 1
try:
    idx_sub_gas2 = names_comp.index('Subcluster Gas 2') + 1
except ValueError:
    idx_sub_gas2 = np.nan
try:
    idx_sub_gas3 = names_comp.index('Sub Gas 3') + 1
except ValueError:
    idx_sub_gas3 = np.nan

arcsec_conv = (180/np.pi) * 3600 / D_L
comp_xy_arcsec = np.zeros((N_comp, 2))
for j in range(1, N_comp+1):
    comp_xy_arcsec[j-1, 0] = components[j]['x_c'] * arcsec_conv
    comp_xy_arcsec[j-1, 1] = components[j]['y_c'] * arcsec_conv

main_gal_xy = comp_xy_arcsec[idx_bcg1-1, :]
sub_gal_xy  = comp_xy_arcsec[idx_bcg3-1, :]
main_gas_xy = comp_xy_arcsec[idx_main_gas-1, :]
sub_gas1_xy = comp_xy_arcsec[idx_sub_gas1-1, :]

sub_gas2_xy = comp_xy_arcsec[idx_sub_gas2-1, :] if not np.isnan(idx_sub_gas2) else [np.nan, np.nan]
sub_gas3_xy = comp_xy_arcsec[idx_sub_gas3-1, :] if not np.isnan(idx_sub_gas3) else [np.nan, np.nan]

print('[BRIDGE] Pipeline variables generated successfully.')

# 2.10: VISUALIZATION SETUP
kpc_per_arcsec = arcsec_to_kpc
xlim_arcsec = [-150, 300]
ylim_arcsec = [-150, 200]

# ========================================================================
#    FIGURE 1: SIMPLE TOPOLOGY
# ========================================================================
fig1 = plt.figure('Fig1_Topology', figsize=(10, 8))
ax = fig1.add_axes([0.08, 0.08, 0.88, 0.88])
ax.set_facecolor('w')

c_main_gas = [0.05, 0.40, 0.65];   c_sub_gas  = [0.60, 0.80, 0.30]
c_bcg_main = [0.00, 0.35, 0.60];   c_bcg_sub  = [0.55, 0.75, 0.25]
c_main_gal = [0.00, 0.35, 0.60];   c_sub_gal  = [0.55, 0.75, 0.25]

theta = np.linspace(0, 2*np.pi, 200)
for i in gas_idx:
    x_c, y_c, r = comp_xy_kpc[i-1, 0], comp_xy_kpc[i-1, 1], components[i]['r_s'] / kpc
    if 'Main Gas' in components[i]['name']:
        ax.fill(x_c + r*np.cos(theta), y_c + r*np.sin(theta), color=c_main_gas, alpha=0.20, ec='k', lw=0.8)
    elif 'Subcluster Gas 1' in components[i]['name']:
        ax.fill(x_c + r*np.cos(theta), y_c + r*np.sin(theta), color=c_sub_gas, alpha=0.15, ec='k', lw=0.8)
    elif 'Subcluster Gas 2' in components[i]['name']:
        ax.fill(x_c + r*np.cos(theta), y_c + r*np.sin(theta), color=c_sub_gas, alpha=0.25, ec='k', lw=0.8)
    elif 'Sub Gas 3' in components[i]['name']:
        ax.fill(x_c + r*np.cos(theta), y_c + r*np.sin(theta), color=c_sub_gas, alpha=0.25, ec='k', lw=0.8)

r_bcg_plot = 20
for i in [idx_bcg1, idx_bcg2, idx_bcg3]:
    x_c, y_c = comp_xy_kpc[i-1, 0], comp_xy_kpc[i-1, 1]
    bcg_color = c_bcg_sub if i == idx_bcg3 else c_bcg_main
    ax.fill(x_c + r_bcg_plot*np.cos(theta), y_c + r_bcg_plot*np.sin(theta), color=bcg_color, alpha=0.90, ec='k', lw=1.5)

main_gal_idx = [i+1 for i, n in enumerate(names_comp) if 'Main Gal' in n]
for i in main_gal_idx:
    ax.plot(comp_xy_kpc[i-1, 0], comp_xy_kpc[i-1, 1], 'o', ms=3, color=c_main_gal, mec='k', mew=0.4)

sub_gal_idx = [i+1 for i, n in enumerate(names_comp) if 'Sub Gal' in n]
for i in sub_gal_idx:
    ax.plot(comp_xy_kpc[i-1, 0], comp_xy_kpc[i-1, 1], 'o', ms=3, color=c_sub_gal, mec='k', mew=0.4)

for i in gas_idx:
    x_c, y_c, r = comp_xy_kpc[i-1, 0], comp_xy_kpc[i-1, 1], components[i]['r_s'] / kpc
    if 'Main Gas' in components[i]['name']:
        ax.text(x_c, y_c + r + 15, 'Main Gas', ha='center', fontsize=9, color='k', fontweight='bold')
    elif 'Subcluster Gas 1' in components[i]['name']:
        ax.text(x_c, y_c + r + 15, 'Sub Gas 1', ha='center', fontsize=9, color='k', fontweight='bold')
    elif 'Subcluster Gas 2' in components[i]['name']:
        ax.text(x_c + r + 25, y_c, 'Sub Gas 2', ha='left', fontsize=9, color='k', fontweight='bold')
    elif 'Sub Gas 3' in components[i]['name']:
        ax.text(x_c - r - 25, y_c, 'Sub Gas 3', ha='right', fontsize=9, color='k', fontweight='bold')

for i, txt in zip([idx_bcg1, idx_bcg2, idx_bcg3], ['BCG1', 'BCG2', 'BCG3']):
    ax.text(comp_xy_kpc[i-1, 0], comp_xy_kpc[i-1, 1]+r_bcg_plot+25, txt, ha='center', fontsize=10, color='k', fontweight='bold')

ax.set_xlim([-600, 1100]); ax.set_ylim([-600, 800])
ax.set_xlabel('x [kpc]', fontsize=12); ax.set_ylabel('y [kpc]', fontsize=12)
ax.grid(True, alpha=0.15)
ax.set_aspect('equal')

bar_len, bar_x, bar_y = 100, 900, -500
ax.plot([bar_x, bar_x+bar_len], [bar_y, bar_y], 'k', lw=2.5)
ax.plot([bar_x, bar_x], [bar_y-5, bar_y+5], 'k', lw=2)
ax.plot([bar_x+bar_len, bar_x+bar_len], [bar_y-5, bar_y+5], 'k', lw=2)
ax.text(bar_x+bar_len/2, bar_y+15, '100 kpc', ha='center', fontsize=9, fontweight='bold', color='k')

fig1.savefig('Fig1_Topology.jpg', dpi=300)
print('[FIGURE 1] Saved successfully.')

# =================================================================
#  FIGURE 2: COMPARISON PLOT
# =================================================================
print('\n[COMPARE] Generating multi-slice gas surface density comparison...')

match_gas_M = np.array([2.0e14, 1.5e13, 1.5e13, -6.8e12]) * Msun
match_gas_a = np.array([565, 505, 90, 70]) * 1e3 * pc
match_gas_x = np.array([190, 525, 670, 670]) * 1e3 * pc
match_gas_y = np.array([90, 120, 170, 170]) * 1e3 * pc

notaper_gas_M = np.array([components[1]['M_total'], components[2]['M_total'], components[3]['M_total'], components[4]['M_total']])
notaper_gas_a = np.array([components[1]['r_s'], components[2]['r_s'], components[3]['r_s'], components[4]['r_s']])
notaper_gas_x = np.array([components[1]['x_c'], components[2]['x_c'], components[3]['x_c'], components[4]['x_c']])
notaper_gas_y = np.array([components[1]['y_c'], components[2]['y_c'], components[3]['y_c'], components[4]['y_c']])

y_slices = [120, 170, 220]
slice_colors = np.array([[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.466, 0.674, 0.188]])
x_slice_kpc = np.linspace(400, 900, 500)
X_m = x_slice_kpc * 1e3 * pc

fig2 = plt.figure('Fig2_GasDensityComparison', figsize=(8.5, 6.5))
ax = fig2.add_subplot(111)
c_ref = [0.3, 0.3, 0.3]

for s in range(len(y_slices)):
    y_val = y_slices[s]
    Y_m = y_val * 1e3 * pc * np.ones_like(X_m)
    Sigma_match = np.zeros_like(X_m)
    Sigma_notaper = np.zeros_like(X_m)

    for i in range(len(match_gas_M)):
        R2_match = (X_m - match_gas_x[i])**2 + (Y_m - match_gas_y[i])**2
        Sigma_match += (match_gas_M[i] * match_gas_a[i]**2) / (np.pi * (R2_match + match_gas_a[i]**2)**2)

        R2_notaper = (X_m - notaper_gas_x[i])**2 + (Y_m - notaper_gas_y[i])**2
        Sigma_notaper += (notaper_gas_M[i] * notaper_gas_a[i]**2) / (np.pi * (R2_notaper + notaper_gas_a[i]**2)**2)

    Sigma_match_msun = Sigma_match * (pc**2) / Msun
    Sigma_notaper_msun = Sigma_notaper * (pc**2) / Msun

    ax.semilogy(x_slice_kpc, Sigma_match_msun, '-', color=slice_colors[s, :], lw=2.5, label=f'y = {y_val} kpc (Famaey Model)')
    ax.semilogy(x_slice_kpc, Sigma_notaper_msun, '--', color=slice_colors[s, :], lw=2.5, label=f'y = {y_val} kpc (Flat positive)')

ax.axvline(670, color=c_ref, ls=':', lw=1.5)
ax.text(670, ax.get_ylim()[0]*2, 'Subcluster Gas Center (670 kpc)', ha='center', va='bottom', fontsize=9)
ax.set_xlabel('X [kpc]', fontsize=13, fontweight='bold')
ax.set_ylabel(r'Gas Surface Density $\Sigma$ [$M_\odot$ / pc$^2$]', fontsize=13, fontweight='bold')
ax.legend(loc='lower left', fontsize=10)
ax.grid(True, color=[0.85, 0.85, 0.85], alpha=0.6, ls=':')
fig2.savefig('Fig2_GasDensityComparison.jpg', dpi=300)
print('[FIGURE 2] Multi-slice plot saved successfully.')

# ========================================================================
# SECTION 3: COMPUTATIONAL GRIDS (ADAPTIVE STRETCHED GRID)
# ========================================================================
u_faces = np.linspace(-1, 1, N_2d + 1)
if stretch_factor == 0:
    x_faces = u_faces * (Box_3d/2); y_faces = u_faces * (Box_3d/2)
else:
    print('[GRID] Building dual-focus adaptive grid (Main + Subcluster)...')
    x_faces = generate_dual_focus_faces(u_faces, Box_3d/2, 0, x_sub_c_kpc*kpc, focus_width)
    y_faces = generate_dual_focus_faces(u_faces, Box_3d/2, 0, y_sub_c_kpc*kpc, focus_width)

dx_fine = np.diff(x_faces); dy_fine = np.diff(y_faces)
x_fine = (x_faces[:-1] + x_faces[1:]) / 2
y_fine = (y_faces[:-1] + y_faces[1:]) / 2

u_z_faces = np.linspace(-1, 1, N_z + 1)
if stretch_factor == 0:
    z_faces = u_z_faces * (Box_3d/2)
else:
    z_faces = generate_dual_focus_faces(u_z_faces, Box_3d/2, 0, 0, focus_width_z)

dz_3d = np.diff(z_faces)
z_axis = (z_faces[:-1] + z_faces[1:]) / 2

theta_x_faces = x_faces / D_L * (180/np.pi) * 3600
theta_y_faces = y_faces / D_L * (180/np.pi) * 3600
theta_x = (theta_x_faces[:-1] + theta_x_faces[1:]) / 2
theta_y = (theta_y_faces[:-1] + theta_y_faces[1:]) / 2

X_FINE, Y_FINE = np.meshgrid(x_fine, y_fine, indexing='ij')

# ========================================================================
# SECTION 4: ANALYTICAL PLUMMER POTENTIALS
# ========================================================================
print('[POISSON] Building vectorized Plummer potential function...')

M_arr  = np.array([components[i]['M_total'] for i in range(1, N_comp+1)])[np.newaxis, np.newaxis, :]
xc_arr = np.array([components[i]['x_c'] for i in range(1, N_comp+1)])[np.newaxis, np.newaxis, :]
yc_arr = np.array([components[i]['y_c'] for i in range(1, N_comp+1)])[np.newaxis, np.newaxis, :]
zc_arr = np.array([components[i]['z_c'] for i in range(1, N_comp+1)])[np.newaxis, np.newaxis, :]
a2_arr = np.array([components[i]['r_s']**2 for i in range(1, N_comp+1)])[np.newaxis, np.newaxis, :]

def Phi_total_pos_func(x, y, z):
    # x, y, z expected to be shape (N_r, N_dirs, 1)
    r2 = (x - xc_arr)**2 + (y - yc_arr)**2 + (z - zc_arr)**2 + a2_arr
    return np.sum(-G * M_arr / np.sqrt(r2), axis=2)

M_total_bary = np.sum(M_arr)
M_true_total_bary = M_total_bary
print(f'  True total baryonic M: {M_true_total_bary/Msun:.3e} Msun')

# ========================================================================
# SECTION 5: MEZZI RAY TRACING
# ========================================================================
R_source_limit = max(D_L, 0.1*pc)
R_box = Box_3d/2

print('[K] Ray tracing directly on 2D planes...')

if R_source_limit <= R_box:
    r_ray = np.logspace(np.log10(0.1*pc), np.log10(R_source_limit), N_r_ray)
else:
    N_r_int = round(N_r_ray * (np.log10(R_box)-np.log10(0.1*pc)) / (np.log10(R_source_limit)-np.log10(0.1*pc)))
    N_r_int = max(N_r_int, 50)
    N_r_ext = N_r_ray - N_r_int
    r_int = np.logspace(np.log10(0.1*pc), np.log10(R_box), N_r_int)
    r_ext = np.logspace(np.log10(R_box), np.log10(R_source_limit), N_r_ext+1)
    r_ray = np.concatenate((r_int, r_ext[1:]))

sum_inv_zeta_global = np.zeros_like(X_FINE)
total_path_length = 0

print(f'[K] Starting line-of-sight ray tracing over {N_z} Z-planes...')
import time
start_t = time.time()
for k in range(N_z):
    if k % 20 == 0 or k == 0 or k == N_z-1:
        print(f'  [K] Processing Z-plane {k+1} of {N_z} ({100*(k+1)/N_z:.1f}%)...')

    Z_k = z_axis[k]
    Z_plane = np.ones_like(X_FINE) * Z_k

    _, Zeta_2D = trace_component_K(
        {'x_c':0, 'y_c':0, 'z_c':0}, Phi_total_pos_func, M_true_total_bary,
        X_FINE, Y_FINE, Z_plane, r_ray, R_box, c, Compliance, N_theta, N_phi
    )

    Zeta_2D = np.maximum(Zeta_2D, np.finfo(float).tiny)

    dz = dz_3d[k]
    sum_inv_zeta_global += (dz / Zeta_2D)
    total_path_length += dz

print(f'[K] Ray tracing complete. Elapsed time: {time.time()-start_t:.2f} seconds.\n')

zeta_2D_fine = total_path_length / sum_inv_zeta_global
print(f'[K] Global K complete. Min zeta = {np.min(zeta_2D_fine):.4e}')

# ========================================================================
# SECTION 6: PROJECTION, RESCALING & LENSING PIPELINE
# ========================================================================
print('[LENS] Projecting densities and building convergence maps...')

Sigma_stars_fine_SI = np.zeros_like(X_FINE)
Sigma_gas_fine_SI   = np.zeros_like(X_FINE)

for j in range(1, N_comp+1):
    comp = components[j]
    if comp['M_total'] == 0: continue

    if comp['type'] == 'Plummer':
        Xs_f = X_FINE - comp['x_c']; Ys_f = Y_FINE - comp['y_c']
        R2_f = Xs_f**2 + Ys_f**2
        Sigma_f = (comp['M_total'] * comp['r_s']**2) / (np.pi * (R2_f + comp['r_s']**2)**2)

        if j in star_idx:
            Sigma_stars_fine_SI += Sigma_f
        else:
            Sigma_gas_fine_SI += Sigma_f

Sigma_stars_fine = Sigma_stars_fine_SI * (pc**2)/Msun
Sigma_gas_fine   = Sigma_gas_fine_SI   * (pc**2)/Msun

# MEZZI STEP 3: TRUE FRAME MASS VIA AREA & GEOMETRY
J_vol = 1.0 / zeta_2D_fine
J_vol[~np.isfinite(J_vol)] = 1
J_vol[J_vol < 0] = 0
J = J_vol**(2/3)

Sigma_true_stars = Sigma_stars_fine * J
Sigma_true_gas   = Sigma_gas_fine * J

lambda_2D = (1.0 / zeta_2D_fine)**(2/3)
lambda_2D[~np.isfinite(lambda_2D)] = 1

DX, DY = np.meshgrid(dx_fine, dy_fine, indexing='ij')
dA_obs_pc2 = (DX/pc) * (DY/pc)

M_obs_stars = np.sum(Sigma_stars_fine * dA_obs_pc2)
M_obs_gas   = np.sum(Sigma_gas_fine   * dA_obs_pc2)
M_obs_total = M_obs_stars + M_obs_gas

M_true_stars = np.sum(Sigma_true_stars * dA_obs_pc2)
M_true_gas   = np.sum(Sigma_true_gas   * dA_obs_pc2)
M_true_total = M_true_stars + M_true_gas
mass_ratio_eta = M_true_total / M_obs_total

print(f'  Observed baryonic: {M_obs_total:.3e} Msun')
print(f'  True baryonic:     {M_true_total:.3e} Msun (eta={mass_ratio_eta:.3f})')

Sigma_obs_fine = Sigma_stars_fine + Sigma_gas_fine
kappa_bary        = Sigma_obs_fine / Sigma_crit
kappa_mass_reveal = (Sigma_true_stars + Sigma_true_gas) / Sigma_crit
kappa_curv_amp    = lambda_2D * kappa_bary

# ========================================================================
# SECTION 7: LENSING SOLVER (UNIFORM PROXY GRID)
# ========================================================================
print('[LENS] Preparing uniform proxy grid for FFT...')

fft_box_half_size = fft_box_half_size_kpc * 1e3 * pc
x_uni = np.linspace(-fft_box_half_size, fft_box_half_size, N_fft)
y_uni = np.linspace(-fft_box_half_size, fft_box_half_size, N_fft)

print(f'[FFT] Cropping uniform grid to +/- {fft_box_half_size_kpc} kpc (Resolution: {2*fft_box_half_size_kpc/N_fft:.2f} kpc/pix)')

theta_x_uni = x_uni / D_L * (180/np.pi) * 3600
theta_y_uni = y_uni / D_L * (180/np.pi) * 3600
dtheta_uni = theta_x_uni[1] - theta_x_uni[0]

models = {
    'bary': {'name': 'Baryonic Baseline', 'kappa': kappa_bary},
    'mass_reveal': {'name': 'Mass Revelation', 'kappa': kappa_mass_reveal},
    'curv_amp': {'name': 'Curvature Amplification', 'kappa': kappa_curv_amp}
}

# Interpolate to uniform grid
F_kappa_bary = RegularGridInterpolator((theta_x, theta_y), kappa_bary, method='linear', bounds_error=False, fill_value=0)
F_kappa_mr   = RegularGridInterpolator((theta_x, theta_y), kappa_mass_reveal, method='linear', bounds_error=False, fill_value=0)
F_kappa_ca   = RegularGridInterpolator((theta_x, theta_y), kappa_curv_amp, method='linear', bounds_error=False, fill_value=0)

T_UNI, T_Y_UNI = np.meshgrid(theta_x_uni, theta_y_uni, indexing='ij')
kappa_bary_uni = F_kappa_bary((T_UNI, T_Y_UNI))
kappa_mr_uni   = F_kappa_mr((T_UNI, T_Y_UNI))
kappa_ca_uni   = F_kappa_ca((T_UNI, T_Y_UNI))

if Sigma_smoothing and smoothing_fwhm_arcsec > 0:
    sigma_arcsec = smoothing_fwhm_arcsec / 2.355
    sigma_pix_uniform = sigma_arcsec / dtheta_uni
    print(f'[LENS] Applying physical PSF smoothing (FWHM={smoothing_fwhm_arcsec:.1f}", sigma={sigma_pix_uniform:.2f} pix)...')

    kappa_bary_uni = gaussian_filter(kappa_bary_uni, sigma=sigma_pix_uniform)
    kappa_mr_uni   = gaussian_filter(kappa_mr_uni, sigma=sigma_pix_uniform)
    kappa_ca_uni   = gaussian_filter(kappa_ca_uni, sigma=sigma_pix_uniform)

    print('[LENS] Interpolating smoothed maps back to adaptive grid...')
    F_kappa_bary_smoothed = RegularGridInterpolator((theta_x_uni, theta_y_uni), kappa_bary_uni, method='linear', bounds_error=False, fill_value=0)
    F_kappa_mr_smoothed   = RegularGridInterpolator((theta_x_uni, theta_y_uni), kappa_mr_uni, method='linear', bounds_error=False, fill_value=0)
    F_kappa_ca_smoothed   = RegularGridInterpolator((theta_x_uni, theta_y_uni), kappa_ca_uni, method='linear', bounds_error=False, fill_value=0)

    T_FINE, T_Y_FINE = np.meshgrid(theta_x, theta_y, indexing='ij')
    kappa_bary        = F_kappa_bary_smoothed((T_FINE, T_Y_FINE))
    kappa_mass_reveal = F_kappa_mr_smoothed((T_FINE, T_Y_FINE))
    kappa_curv_amp    = F_kappa_ca_smoothed((T_FINE, T_Y_FINE))
    print('[LENS] Smoothing complete.')

# ========================================================================
# SECTION 8: VISUALIZATION
# ========================================================================

# FIGURE 3: BARYONIC TOPOLOGY MAP
c_main_gal  = [0.100, 0.350, 0.600]; c_sub_gal   = [0.050, 0.250, 0.450]
c_main_gas  = [0.850, 0.300, 0.100]; c_sub_gas   = [0.950, 0.550, 0.150]

fig3 = plt.figure('Fig3_BaryonicTopologyInGrid', figsize=(9, 7.8))
ax = fig3.add_axes([0.10, 0.10, 0.78, 0.86])
ax.set_facecolor([0.98, 0.98, 0.98])

comp = {}
for j in range(1, N_comp+1):
    if components[j]['M_total'] == 0: continue
    comp[j] = {
        'name': components[j]['name'],
        'x_c': components[j]['x_c'] / D_L * (180/np.pi) * 3600,
        'y_c': components[j]['y_c'] / D_L * (180/np.pi) * 3600,
        'r_s': components[j]['r_s'] / D_L * (180/np.pi) * 3600,
        'M': components[j]['M_total'] / Msun,
        'is_gas': 'Gas' in components[j]['name']
    }

all_x = [v['x_c'] for v in comp.values()]
all_y = [v['y_c'] for v in comp.values()]
all_r = [v['r_s'] for v in comp.values()]
x_min_comp, x_max_comp = min(np.array(all_x)-np.array(all_r)), max(np.array(all_x)+np.array(all_r))
y_min_comp, y_max_comp = min(np.array(all_y)-np.array(all_r)), max(np.array(all_y)+np.array(all_r))

zoom_factor = 1.4
x_cen = (x_min_comp + x_max_comp)/2; y_cen = (y_min_comp + y_max_comp)/2
x_half = (x_max_comp - x_min_comp)/2 * zoom_factor; y_half = (y_max_comp - y_min_comp)/2 * zoom_factor

xlim_new = [x_cen - x_half, x_cen + x_half]
ylim_new = [y_cen - y_half, y_cen + y_half]

for i in range(len(theta_x_faces)):
    if xlim_new[0] <= theta_x_faces[i] <= xlim_new[1]:
        ax.plot([theta_x_faces[i], theta_x_faces[i]], ylim_new, color=[0.88, 0.88, 0.92], lw=0.25)
for j in range(len(theta_y_faces)):
    if ylim_new[0] <= theta_y_faces[j] <= ylim_new[1]:
        ax.plot(xlim_new, [theta_y_faces[j], theta_y_faces[j]], color=[0.88, 0.88, 0.92], lw=0.25)

theta_circle = np.linspace(0, 2*np.pi, 300)
for cg in comp.values():
    if 'Gal' in cg['name'] and 'Gas' not in cg['name']:
        ax.plot(cg['x_c'], cg['y_c'], '.', color=[0.4, 0.4, 0.8], ms=6)
        continue

    x_circ = cg['x_c'] + cg['r_s'] * np.cos(theta_circle)
    y_circ = cg['y_c'] + cg['r_s'] * np.sin(theta_circle)

    if cg['is_gas']:
        for r in np.linspace(cg['r_s'], 0, 5):
            x_r = cg['x_c'] + r * np.cos(theta_circle)
            y_r = cg['y_c'] + r * np.sin(theta_circle)
            alpha_r = 0.05 + 0.20*(1 - r/cg['r_s'])
            ax.fill(x_r, y_r, color=c_main_gas if cg['name'] == 'Main Gas' else c_sub_gas, alpha=alpha_r, ec='none')
        ax.plot(x_circ, y_circ, '-', color=c_main_gas if cg['name'] == 'Main Gas' else c_sub_gas, lw=1.0)
    else:
        color = c_main_gal if cg['name'] == 'BCG 1' else (c_sub_gal if cg['name'] == 'BCG 3' else [0.5, 0.5, 0.5])
        ax.fill(x_circ, y_circ, color=color, alpha=0.35, ec=color, lw=1.8)
        x_core = cg['x_c'] + 0.3*cg['r_s'] * np.cos(theta_circle)
        y_core = cg['y_c'] + 0.3*cg['r_s'] * np.sin(theta_circle)
        ax.fill(x_core, y_core, color='w', alpha=0.4, ec='none')

ax.set_xlim([-150, 300])
ax.set_ylim([-150, 200])
ax.set_xlabel(r'$\theta_x$ [arcsec]', fontsize=13)
ax.set_ylabel(r'$\theta_y$ [arcsec]', fontsize=13)
ax.set_title('Baryonic Topology --- Plummer Components & Lensing Peaks', fontsize=14, fontweight='bold')

info_text = f"Mass Budget\nMstars = {M_obs_stars:.2e} Msun\nMgas = {M_obs_gas:.2e} Msun\nMtrue = {M_true_total:.2e} Msun\neta = {mass_ratio_eta:.3f}"
ax.text(0.72, 0.3, info_text, transform=fig3.transFigure, fontsize=9, family='monospace', va='top', ha='left', bbox=dict(boxstyle='round', facecolor=[1, 1, 0.97], edgecolor=[0.5, 0.5, 0.5]))

fig3.savefig('Fig3_BaryonicTopologyInGrid.jpg', dpi=300)
print('[FIGURE 3] Saved successfully.')

# FIGURE 4: MASS REVELATION CONVERGENCE κ

# Prepare 2D grids (Crucial for non-uniform contour plotting)
T_X, T_Y = np.meshgrid(theta_x, theta_y, indexing='ij')
X_kpc_plot = T_X * arcsec_to_kpc
Y_kpc_plot = T_Y * arcsec_to_kpc

# Clip and mask invalid values to prevent contour gaps
kc = np.clip(kappa_mass_reveal, 1e-4, 10)
log_kc = np.log10(kc)
log_kc = np.ma.masked_invalid(log_kc)

kappa_bary_m = np.ma.masked_invalid(kappa_bary)
kappa_mr_m = np.ma.masked_invalid(kappa_mass_reveal)

fig4 = plt.figure('Fig4_MassRevelationKappa', figsize=(11, 8.5))
ax = fig4.add_axes([0.08, 0.10, 0.70, 0.80])

log_lo, log_hi = -0.3, 0.5
key_colors = np.array([[0.02, 0.05, 0.20], [0.10, 0.25, 0.55], [0.25, 0.60, 0.80], [0.60, 0.90, 0.70], [1.00, 1.00, 1.00]])
cmap_kappa = LinearSegmentedColormap.from_list('cmap_kappa', key_colors[::-1], N=256)

# Use the 2D coordinates explicitly
mesh = ax.pcolormesh(X_kpc_plot, Y_kpc_plot, log_kc, shading='auto', cmap=cmap_kappa, vmin=log_lo, vmax=log_hi)

# Baryonic baseline contours
# Baryonic baseline contours
bary_levels = [0.05, 0.10, 0.20]
if np.max(kappa_bary_m) >= min(bary_levels):
    cs = ax.contour(X_kpc_plot, Y_kpc_plot, kappa_bary_m, levels=bary_levels, colors=[(0.7,0.7,0.7)], linestyles='--', linewidths=0.8, corner_mask=False)
    ax.clabel(cs, inline=False, fontsize=8, fmt='%.2f')

kappa_levels = [0.2, 0.5, 1.0]
cs2 = ax.contour(X_kpc_plot, Y_kpc_plot, kappa_mr_m, levels=kappa_levels, colors=[(0.1,0.1,0.1)], linewidths=1.2, corner_mask=False)
ax.clabel(cs2, inline=False, fontsize=10, fmt='%.1f')

# Add legend
h_leg_true_k, = ax.plot([], [], '-', color=(0.1, 0.1, 0.1), lw=1.2, label=r'True frame $\kappa$')
h_leg_bary, = ax.plot([], [], '--', color=(0.7, 0.7, 0.7), lw=1.0, label=r'Baryonic baseline $\kappa$')
ax.legend(handles=[h_leg_true_k, h_leg_bary], loc='upper center', fontsize=9, frameon=True, edgecolor=(0.6, 0.6, 0.6))

ax.set_xlim([-150*arcsec_to_kpc, 300*arcsec_to_kpc])
ax.set_ylim([-150*arcsec_to_kpc, 200*arcsec_to_kpc])
ax.set_xlabel('x [kpc]', fontsize=13)
ax.set_ylabel('y [kpc]', fontsize=13)
ax.set_title(r'Mass Revelation Convergence $\kappa_{\rm Mezzi}$', fontsize=14, fontweight='bold')
ax.set_axisbelow(False)

cax = fig4.add_axes([0.80, 0.10, 0.025, 0.80])
cbar = fig4.colorbar(mesh, cax=cax)
cbar.set_label(r'$\log_{10} \kappa$', fontsize=12)
cbar.set_ticks(np.arange(-0.5, 0.51, 0.1))

# ========================================================================
#  INSET: BCG3 Zoom
# ========================================================================
# Hardcode BCG3 position in kpc based on catalog expectations
bcg3_kpc = np.array([820.0, 220.0])
zoom_half = 30
zoom_xlim = [bcg3_kpc[0] - zoom_half, bcg3_kpc[0] + zoom_half]
zoom_ylim = [bcg3_kpc[1] - zoom_half, bcg3_kpc[1] + zoom_half]

# Draw zoom rectangle on main axis
ax.add_patch(Rectangle((zoom_xlim[0], zoom_ylim[0]), zoom_xlim[1]-zoom_xlim[0], zoom_ylim[1]-zoom_ylim[0],
                       edgecolor=[0.85, 0.20, 0.20], facecolor='none', lw=1.5))

# Create inset (repositioned to fit fully inside the main box)
ax_inset = fig4.add_axes([0.53, 0.15, 0.24, 0.24])
ax_inset.pcolormesh(X_kpc_plot, Y_kpc_plot, log_kc, shading='auto', cmap=cmap_kappa, vmin=log_lo, vmax=log_hi)

# Contours inside inset with labels
if np.max(kappa_bary_m) >= min(bary_levels):
    cs_inset_bary = ax_inset.contour(X_kpc_plot, Y_kpc_plot, kappa_bary_m, levels=bary_levels, colors=[(0.7,0.7,0.7)], linestyles='--', linewidths=0.7)
    ax_inset.clabel(cs_inset_bary, inline=False, fontsize=7, fmt='%.2f')

cs_inset_mr = ax_inset.contour(X_kpc_plot, Y_kpc_plot, kappa_mr_m, levels=kappa_levels, colors=[(0.1,0.1,0.1)], linewidths=1.0)
ax_inset.clabel(cs_inset_mr, inline=False, fontsize=8, fmt='%.1f')

ax_inset.set_xlim(zoom_xlim)
ax_inset.set_ylim(zoom_ylim)
ax_inset.plot(bcg3_kpc[0], bcg3_kpc[1], 'r+', ms=10, mew=1.5)
ax_inset.set_title('BCG3 Zoom (60 kpc)', fontsize=9, fontweight='bold')
ax_inset.set_xlabel('x [kpc]', fontsize=8)
ax_inset.set_ylabel('y [kpc]', fontsize=8)
ax_inset.tick_params(labelsize=8)


fig4.savefig('Fig4_MassRevelationKappa.jpg', dpi=300)
print('[FIGURE 4] Saved successfully.')

# FIGURE 5: COMPONENT-LEVEL MASS COMPARISON
fig5 = plt.figure('Fig5_MassComparison_Spatial', figsize=(9.5, 6.5))
ax_main = fig5.add_axes([0.10, 0.12, 0.75, 0.82])

numComp = N_comp
M_obs_all = np.zeros(numComp + 1)
M_true_all = np.zeros(numComp + 1)
X_pos_all = np.zeros(numComp + 1)
comp_names_all = [None] * (numComp + 1)

F_Jmap = RegularGridInterpolator((x_fine/(pc*1e3), y_fine/(pc*1e3)), J, method='linear', bounds_error=False, fill_value=1.0)

print('\n[FIGURE 5] Calculating component masses...')
for j in range(1, numComp+1):
    comp = components[j]
    M_obs = abs(comp['M_total']) / Msun
    if M_obs == 0: continue

    is_compact_galaxy = 'Gal' in comp['name'] and 'Gas' not in comp['name']
    if is_compact_galaxy:
        J_at_comp = F_Jmap([[comp['x_c']/(pc*1e3), comp['y_c']/(pc*1e3)]])[0]
        M_true_val = M_obs * J_at_comp
    else:
        Xs_f = X_FINE - comp['x_c']; Ys_f = Y_FINE - comp['y_c']
        R2_f = Xs_f**2 + Ys_f**2
        Sigma_obs_j_SI = (comp['M_total'] * comp['r_s']**2) / (np.pi * (R2_f + comp['r_s']**2)**2)
        Sigma_obs_j_Msun_pc2 = Sigma_obs_j_SI * (pc**2 / Msun)
        M_true_val = np.sum(Sigma_obs_j_Msun_pc2 * J * dA_obs_pc2)

    M_true_val = max(M_true_val, M_obs)
    M_obs_all[j-1] = M_obs; M_true_all[j-1] = M_true_val
    X_pos_all[j-1] = comp['x_c'] / (pc*1e3)
    comp_names_all[j-1] = comp['name']

M_obs_all[-1] = M_obs_total; M_true_all[-1] = M_true_total
valid_comp_idx = M_obs_all[:-1] > 0
X_pos_all[-1] = np.mean(X_pos_all[:-1][valid_comp_idx])
comp_names_all[-1] = 'Overall Cluster'

valid_idx = (M_obs_all > 0) & (M_true_all > 0)
M_obs_valid = M_obs_all[valid_idx]; M_true_valid = M_true_all[valid_idx]
X_pos_valid = X_pos_all[valid_idx]; comp_names_valid = [n for i, n in enumerate(comp_names_all) if valid_idx[i]]

sort_idx = np.argsort(X_pos_valid)
X_pos_sorted = X_pos_valid[sort_idx]
log_M_obs_sorted = np.log10(M_obs_valid[sort_idx])
log_M_true_sorted = np.log10(M_true_valid[sort_idx])
comp_names_sorted = [comp_names_valid[i] for i in sort_idx]
correction_sorted = M_true_valid[sort_idx] / M_obs_valid[sort_idx]

cmap = plt.cm.turbo
corr_min_log, corr_max_log = 0, 1.7
log_correction = np.log10(correction_sorted)

for g in range(len(log_M_obs_sorted)):
    norm_val = (log_correction[g] - corr_min_log) / (corr_max_log - corr_min_log)
    idx = max(1, min(256, round(norm_val * 255)))
    pt_color = cmap(idx)
    ax_main.plot([X_pos_sorted[g], X_pos_sorted[g]], [log_M_obs_sorted[g], log_M_true_sorted[g]], '-', color=pt_color, alpha=0.15, lw=6)
    ax_main.plot([X_pos_sorted[g], X_pos_sorted[g]], [log_M_obs_sorted[g], log_M_true_sorted[g]], '-', color=pt_color, lw=2.5)

h_obs = ax_main.scatter(X_pos_sorted, log_M_obs_sorted, 80, color=[0.85, 0.85, 0.85], edgecolors=[0.15, 0.15, 0.15], label='Observed Mass')
h_true = ax_main.scatter(X_pos_sorted, log_M_true_sorted, 80, log_correction, cmap=cmap, marker='^', edgecolors=[0.15, 0.15, 0.15], label='True Mass', vmin=corr_min_log, vmax=corr_max_log)

cbar = fig5.colorbar(h_true, ax=ax_main)
cbar.set_label(r'M_{true}/M_{obs}', fontsize=18, fontweight='bold')
cbar.set_ticks([np.log10(1), np.log10(5), np.log10(10), np.log10(20), np.log10(50)])
cbar.set_ticklabels(['1x', '5x', '10x', '20x', '50x'])
cbar.ax.tick_params(labelsize=10)

for g, name in enumerate(comp_names_sorted):
    if 'Gas' in name or 'BCG' in name or name == 'Overall Cluster':
        y_pos = log_M_true_sorted[g]
        if name == 'Subcluster Gas 2': y_pos += 0.15
        if name == 'Overall Cluster': y_pos -= 0.3
        ax_main.text(X_pos_sorted[g] + 25, y_pos, name, fontsize=9, color='k', fontweight='bold')

ax_main.set_xlabel('Component X position [kpc]', fontsize=18, fontweight='bold')
ax_main.set_ylabel(r'$log_{10}(M/M_{\odot})$', fontsize=18, fontweight='bold')

# Add legend to match MATLAB version
ax_main.legend(handles=[h_obs, h_true], loc='upper left', fontsize=11, frameon=True, edgecolor=[0.3, 0.3, 0.3])

ax_main.set_xlim([-670, 1250])
ax_main.set_xticks(np.arange(-600, 1201, 200))
ax_main.tick_params(labelsize=10)
ax_main.grid(axis='y', alpha=0.15)
ax_main.set_axisbelow(False)

all_y_values = np.concatenate((log_M_obs_sorted, log_M_true_sorted))
y_min = np.min(all_y_values)
y_max = np.max(all_y_values)
y_pad = (y_max - y_min) * 0.08
ax_main.set_ylim([y_min - y_pad, y_max + y_pad])

fig5.savefig('Fig5_MassComparison_Spatial.jpg', dpi=300)
print('[FIGURE 5] Saved successfully.')

# ========================================================================
#  SECTION 9: SAVE RESULTS
# ========================================================================
print('\n[SAVE] Compiling and saving results to .mat file...')

Results = {
    'Config': {
        'Compliance': Compliance, 'Sigma_smoothing': Sigma_smoothing, 'smoothing_fwhm_arcsec': smoothing_fwhm_arcsec,
        'Box_3d': Box_3d, 'N_2d': N_2d, 'N_z': N_z, 'stretch_factor': stretch_factor,
        'N_fft': N_fft, 'fft_box_half_size_kpc': fft_box_half_size_kpc,
        'N_theta': N_theta, 'N_phi': N_phi, 'N_r_ray': N_r_ray
    },
    'Cosmology': {
        'z_lens': z_lens, 'D_L': D_L, 'Sigma_crit': Sigma_crit, 'arcsec_to_kpc': arcsec_to_kpc
    },
    'MassModel': {
        'components': components, 'star_idx': star_idx, 'gas_idx': gas_idx,
        'comp_xy_kpc': comp_xy_kpc, 'comp_xy_arcsec': comp_xy_arcsec,
        'M_main_smooth': M_main_smooth, 'M_sub_smooth': M_sub_smooth
    },
    'Grid': {
        'x_fine': x_fine, 'y_fine': y_fine, 'z_axis': z_axis,
        'theta_x': theta_x, 'theta_y': theta_y,
        'dx_fine': dx_fine, 'dy_fine': dy_fine, 'dz_3d': dz_3d
    },
    'Mezzi': {
        'zeta_2D_fine': zeta_2D_fine, 'J': J, 'lambda_2D': lambda_2D
    },
    'Lensing': {
        'Sigma_stars_fine': Sigma_stars_fine, 'Sigma_gas_fine': Sigma_gas_fine,
        'Sigma_true_stars': Sigma_true_stars, 'Sigma_true_gas': Sigma_true_gas,
        'M_obs_total': M_obs_total, 'M_true_total': M_true_total, 'mass_ratio_eta': mass_ratio_eta,
        'kappa_bary': kappa_bary, 'kappa_mass_reveal': kappa_mass_reveal,
        'kappa_curv_amp': kappa_curv_amp, 'models': models
    }
}

savemat('Mezzi_BulletCluster_Results.mat', {'Results': Results})
print('[SAVE] Results successfully saved to Mezzi_BulletCluster_Results.mat\n')
