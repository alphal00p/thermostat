# thermostat
Simple standalone Python implementation of Local Unitarity for demonstration/pedagogicl purposes only.

In particular this should serve as a basis for investigating applicability of Local Unitarity to pQFT at finite temperature and finite chemical potential.

## Usage

More detailed usage instructions may be given at a later stage, but for now, we limit ourselves to suggesting to run the following help command:

```python
python3 thermostat.py --help
usage: Triangler [-h] [--verbosity {debug,info,critical}] [--parameterisation {cartesian,spherical}] [--improved_ltd] [--integrand_implementation {python,rust}] [--multi_channeling] [--phase {real,imag}] [--delta DELTA] [--sigma SIGMA] [--m_1 M_1]
                 [--m_2 M_2] [--mu_r MU_R] [--m_uv M_UV] [-p P P P P] [-q Q Q Q Q] [--topology {triangle,bubble}] [--epsilon_expansion_term {0,-1,2}]
                 {inspect,integrate,plot,analytical_result} ...

options:
  -h, --help            show this help message and exit
  --verbosity {debug,info,critical}, -v {debug,info,critical}
                        Set verbosity level
  --parameterisation {cartesian,spherical}, -param {cartesian,spherical}
                        Parameterisation to employ.
  --improved_ltd        Use improved LTD expression which does not suffer from numerical instabilities.
  --integrand_implementation {python,rust}, -ii {python,rust}
                        Integrand implementation selected. Default = python
  --multi_channeling, -mc
                        Consider a multi-channeled integrand.
  --phase {real,imag}   Phase to compute. Default = real GeV
  --delta DELTA         Delta. Default = 0.5 GeV
  --sigma SIGMA         Dimensionless sigma for H-function. Default = 1.0
  --m_1 M_1             First mass. Default = 0.01 GeV
  --m_2 M_2             Second mass. Default = 0.02 GeV
  --mu_r MU_R           Renormalization scale. Default = 0.01 GeV
  --m_uv M_UV           UV regularisation mass scale. Default = 0.01 GeV
  -p P P P P            First external. Default = [0.005, 0.0, 0.0, 0.005] GeV
  -q Q Q Q Q            Second external. Default = [0.005, 0.0, 0.0, -0.005] GeV
  --topology {triangle,bubble}, -topology {triangle,bubble}
                        Selected topology. Default = bubble
  --epsilon_expansion_term {0,-1,2}, -eps {0,-1,2}
                        Selected coefficient of the d-dimensional epsilon expansion to consider. Default = 0

commands:
  {inspect,integrate,plot,analytical_result}
                        Various commands available
    inspect             Inspect evaluation of a sample point of the integration space.
    integrate           Integrate the loop amplitude.
    plot                Plot the integrand.
    analytical_result   Evaluate the analytical result for the amplitude.
```

And running the following benchmark integrations can validate the installation:

* a) Integration of a simple finite triangle scalar integral (with not threshold nor UV or IR divergences):
```python3
python3 thermostat.py -t triangle -p 0.5 0 0 0.5 -q 0.5 0 0 -0.5 --m_1 1.0 --m_2 5.0 --improved_ltd integrate --integrator naive --points_per_iteration 100000
[...]
| > Max weight encountered = 5.18115e-04 at xs = [1.8233746758977010e-01 3.0375392620277664e-03 8.1484303838971506e-01]
| > Central value : +1.2731465667017113e-04   +/- 1.45e-07     (0.114%)
| >     vs target : +1.2707591731146093e-04   Δ = +2.39e-07    (0.188%  = 1.64σ)
```

* b) Integration of a two-point bubble scalar integral with threshold and UV divergences (and no IR divergences as we require $p^2 \ne 0$):
```python3
python3 thermostat.py -t bubble -p 1 0 0 0 --m_uv 0.5 --m_1 0.4 --m_2 0.1 --phase real --improved_ltd integrate --integrator vegas --points_per_iteration 10000
[...]
INFO <module> l.272 t=2023-12-21,01:13:40.284 > --------------------------------------------------------------------------------
INFO <module> l.273 t=2023-12-21,01:13:40.284 > Integration with settings below completed in 4.02s:
| verbosity                     : 'info'
| parameterisation              : 'spherical'
| improved_ltd                  : True
| integrand_implementation      : 'python'
| multi_channeling              : False
| phase                         : 'real'
| delta                         : 0.5
| sigma                         : 1.0
| m_1                           : 0.4
| m_2                           : 0.1
| mu_r                          : 0.01
| m_uv                          : 0.5
| p                             : [1.0, 0.0, 0.0, 0.0]
| q                             : [0.005, 0.0, 0.0, -0.005]
| topology                      : 'bubble'
| epsilon_expansion_term        : 0
| command                       : 'integrate'
| n_iterations                  : 10
| points_per_iteration          : 10000
| integrator                    : 'vegas'
| n_cores                       : 1
| seed                          : None
|
| > Integration result after 186726 evaluations in 3.97 CPU-s (21.3 µs / eval)
| > Max weight encountered = -1.66154e-01 at xs = [5.0000000000000033e-01 5.0000000000000033e-01 5.0000000000000033e-01]
| > Central value : -4.2449519793325646e-02   +/- 1.80e-05     (-0.042%)
| >     vs target : -4.2445835040643413e-02   Δ = -3.68e-06    (0.009%  = 0.20σ)
INFO <module> l.276 t=2023-12-21,01:13:40.284 > --------------------------------------------------------------------------------
```