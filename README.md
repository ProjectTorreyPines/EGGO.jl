# EGGO.jl

EGGO.jl is a Julia package for equilibrium reconstruction and plasma physics analysis using neural networks. It provides functionality for predicting plasma equilibria from diagnostic measurements and boundary data.

## Installation

```julia
using Pkg
Pkg.add("EGGO")
```

## Usage Examples

### Basic Equilibrium Reconstruction from Boundary Data

```julia
using EGGO
using EGGO.IMAS

# Set up parameters
nb = 101
nw = 129
b0 = 2.0
r0 = 1.6955
pend = 0.1

# Example boundary and profile data (from test_efit01)
Rb_target = [1.09559171, 1.0997446376, ...] # Boundary R coordinates
Zb_target = [-0.05000000074505806, 0.040000000596046456, ...] # Boundary Z coordinates
ffp_target = [...]  # F*F' profile data
pp_target = [...]   # Pressure profile data

# Get model components
model_name = :d3d_efit01
green = EGGO.get_greens_function_tables(model_name)
basis_functions = EGGO.get_basis_functions(model_name)
basis_functions_1d, bf1d_itp = EGGO.get_basis_functions_1d(model_name)
wall = EGGO.get_wall(model_name)
NNmodel = EGGO.get_model(model_name)

# Fit profiles
pp_fit, ffp_fit = EGGO.fit_ppffp(pp_target, ffp_target, basis_functions_1d)

# Predict equilibrium
Jt, psirz, Ip = EGGO.predict_model_from_boundary(
    Rb_target, Zb_target, pp_fit, ffp_fit, NNmodel, 
    green, basis_functions, nothing, 0.0, false
)

# Extract key quantities
Ψaxis, Raxis, Zaxis, Ψbnd, ffp, pp = EGGO.get_ΨaxisΨbndffppp(
    psirz, green, basis_functions, basis_functions_1d, 
    bf1d_itp, wall, pp_fit, ffp_fit
)

# Create IMAS data structure
dd = IMAS.dd()
eqt = resize!(dd.equilibrium.time_slice)
EGGO.fill_eqt(eqt, psirz, green, wall, pp, ffp, b0, r0, pend, Ψbnd, Ψaxis, Raxis, Zaxis)
```

### Equilibrium Reconstruction from Coil Currents

```julia
using EGGO

# Example coil current data (from test_efit01_coils)
ffp_target = [-3.05929054, -2.99817803, ...]  # F*F' profile
pp_target = [-292086.832, -291584.589, ...]   # Pressure profile
fcurrt_target = [...]  # F-coil currents
ecurrt_target = zeros(6)  # E-coil currents

# Get model for coil-based reconstruction
model_name = :d3d_efit01efit02cake02_coils
green = EGGO.get_greens_function_tables(model_name)
basis_functions = EGGO.get_basis_functions(model_name)
basis_functions_1d, bf1d_itp = EGGO.get_basis_functions_1d(model_name)
wall = EGGO.get_wall(model_name)
NNmodel = EGGO.get_model(model_name)

# Fit profiles
pp_fit, ffp_fit = EGGO.fit_ppffp(pp_target, ffp_target, basis_functions_1d)

# Predict from coil currents
Jt, psirz, Ip = EGGO.predict_model_from_coils(
    pp_fit, ffp_fit, ecurrt_target, fcurrt_target, 
    NNmodel, green, basis_functions
)
```

### Free-Boundary Reconstruction with Diagnostics

```julia
using EGGO

# Example diagnostic data
shot = 168830
fwtmp2 = [1.0, 1.0, 1.0, ...]  # Mirnov coil weights
fwtsi = [1.0, 1.0, 1.0, ...]   # MSE weights
expsi = [-0.403, 0.061, ...]    # MSE measurements
expmp2 = [0.575, 0.578, ...]    # Mirnov measurements
fcurrt = [-249457.5, -68876.1, ...]  # F-coil currents
ecurrt = [-41790.4, -40929.4, ...]   # E-coil currents
Ip = 1.26e6  # Plasma current

# Get free-boundary model
model_name = :d3d_cakenn_free
green = EGGO.get_greens_function_tables(model_name)
basis_functions = EGGO.get_basis_functions(model_name)
basis_functions_1d, bf1d_itp = EGGO.get_basis_functions_1d(model_name)
wall = EGGO.get_wall(model_name)
NNmodel = EGGO.get_model(model_name)
NNmodel1D = EGGO.get_model1d(model_name)

# Predict equilibrium and profiles
y_psi, y_1d = EGGO.predict_psipla_free(
    shot, expsi, fwtsi, expmp2, fwtmp2, fcurrt, ecurrt, Ip,
    NNmodel, NNmodel1D, green, basis_functions
)

# Calculate boundary
Rb, Zb = EGGO.calculate_boundary(y_psi, fcurrt, ecurrt, green, basis_functions, wall)
```

### Kinetic Profile Prediction

```julia
# Thomson scattering data
r_tom = [1.954, 1.924, 1.890, ...]  # Radial positions
z_tom = [-0.067, -0.065, -0.061, ...]  # Vertical positions  
ne_tom = [5.24e19, 5.07e19, 4.99e19, ...]  # Electron density
Te_tom = [2388.3, 2541.4, 2617.2, ...]     # Electron temperature

# CER data
r_cer = [1.9, 2.0, 2.1]  # CER radial positions
z_cer = [0.0, 0.0, 0.0]  # CER vertical positions
nc_cer = [1.e19, 0.5e19, 0.25e19]  # Carbon density

# Predict kinetic profiles
y_ne, y_Te, y_nc = EGGO.predict_kinetic(
    y_psi[:,1], r_tom, z_tom, ne_tom, Te_tom,
    r_cer, z_cer, nc_cer, fcurrt, ecurrt,
    green, wall, basis_functions, bf1d_itp
)
```

### Processing Time Series Data

```julia
using EGGO
using EGGO.IMAS

# Load IMAS data from HDF5
dd = IMAS.hdf2imas("./dd_200000.h5")

# Set up model
model_name = :d3d_cakenn_free
green = EGGO.get_greens_function_tables(model_name)
basis_functions = EGGO.get_basis_functions(model_name)
basis_functions_1d, bf1d_itp = EGGO.get_basis_functions_1d(model_name)
wall = EGGO.get_wall(model_name)
NNmodel = EGGO.get_model(model_name)
NNmodel1D = EGGO.get_model1d(model_name)

# Process time series
times = dd.magnetics.time
for t in times
    # Extract vectors from IMAS data
    shot, expsi, fwtsi, expmp2, fwtmp2, fcurrt, ecurrt, Ip = 
        EGGO.get_vectors_from_dd(dd, t, green)
    
    # Predict equilibrium
    y_psi, y1d = EGGO.predict_psipla_free(
        shot, expsi, fwtsi, expmp2, fwtmp2, fcurrt, ecurrt, Ip,
        NNmodel, NNmodel1D, green, basis_functions
    )
    
    # Fit profiles if needed
    psirz, pp_fit, ffp_fit, ne, Te, nc, Ti, Vt = EGGO.fit_profiles(
        y_psi, y1d, fcurrt, ecurrt, green, basis_functions, 
        basis_functions_1d, wall
    )
    
    # Calculate boundary if needed
    Rb, Zb = EGGO.calculate_boundary(y_psi, fcurrt, ecurrt, green, basis_functions, wall)
end
```

## Available Models

EGGO.jl includes several pre-trained models for different tokamaks and reconstruction scenarios:

- `:d3d_efit01` - DIII-D fixed-boundary 
- `:d3d_efit01efit02cake02` - DIII-D fixed-boundary multi-model ensemble 
- `:d3d_efit01efit02cake02_coils` - DIII-D fix-boundary multi-model ensemble with coil currents
- `:d3d_cakenn_free` :- DIII-D free-boundary reconstruction

## Testing

Run the test suite to verify installation:

```julia
using Pkg
Pkg.test("EGGO")
```

The test suite includes examples for:
- Fixed-boundary reconstruction (`test_efit01`)
- Coil-based reconstruction (`test_efit01_coils`) 
- Multi-model reconstruction (`test_efit01efit02cake02`)
- Free-boundary reconstruction with diagnostics (`test_cakenn_free`)

See `test/runtests.jl` for detailed examples and usage patterns.
