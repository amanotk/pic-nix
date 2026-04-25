# Foot Example

This setup models the dynamics of a collisionless shock transition layer with
a three-component homogeneous plasma with the periodic boundary condition in
all three directions. The system consists of the reflected ions, the incoming
core (upstream) ions, and the background electrons, all represented by
isotropic Maxwellian distributions in the rest frame of each component.
The simulation frame corresponds to the rest frame of electrons.

Three scenarios are included:
- **alfven**: Ion beam propagating parallel to the ambient magnetic field
  generating Alfven waves via the cyclotron resonance
  (Hoshino & Terasawa, 1985).
- **buneman**: Electrostatic ion-electron beam (Buneman) instability.
- **weibel**: Perpendicular shock with filamentation-type instabilities.


# Physical Parameters

The following parameters should be defined in the configuration file:
- `cc` : speed of light $c$
- `wp` : electron plasma frequency $\omega_{pe}$
- `mime` : ion-to-electron mass ratio $m_i/m_e$
- `ush` : Four-shock speed in units of the speed of light $U_{sh}/c$
- `theta` : polar angle of the ambient magnetic field with respect to the x-axis $\theta$
- `phi` : azimuthal angle of the ambient magnetic field with respect to the x-axis $\phi$
- `sigma` : electron cyclotron-to-plasma frequency squared $\sigma = \Omega_{ce}^2/\omega_{pe}^2$
- `alpha` : density of the reflected ion beam normalized to the total density $\alpha = n_r/n_0$
- `betae` : electron plasma beta $\beta_e = 2 v_{th,e}^2/V_{A,e}^2$
- `betai` : core ion plasma beta $\beta_i = 2 v_{th,i}^2/V_{A,i}^2$
- `betar` : reflected ion plasma beta $\beta_r = 2 v_{th,r}^2/V_{A,i}^2$

The electron charge-to-mass ratio is assumed to be unity $|e|/m_e = 1$,
which gives the ambient magnetic field $B_0 = c \omega_{pe} \sqrt{\sigma}$.  
The three components of the ambient magnetic field are then given by
```math
\begin{aligned}
B_{0,x} &= B_0 \cos \theta \\
B_{0,y} &= B_0 \sin \theta \cos \phi \\
B_{0,z} &= B_0 \sin \theta \sin \phi
\end{aligned}
```

Note that the electron velocity in the shock rest frame is written as
$V_e = -(1 - 2\alpha) V_{sh}$ where
$V_{sh} = U_{sh} / (1 + U_{sh}^2/c^2)^{1/2}$ is the three shock speed
calculated from the four shock speed.
The density ratio $\alpha$ is defined in the shock rest frame, which we need
to convert to the simulation frame to yield
```math
\alpha' = \alpha \frac{1 + (1 - 2\alpha) V_{sh}^2}{1 - (1 - 2\alpha)^2 V_{sh}^2}
```
Similarly, the core and reflected ion drift velocities are given by
```math
\begin{aligned}
  V_i' = \frac{-2 \alpha V_{sh}}{1 - (1 - 2\alpha) V_{sh}^2}\\
  V_r' = \frac{+2 (1-\alpha) V_{sh}}{1 + (1 - 2\alpha) V_{sh}^2}
\end{aligned}
```
These drift velocities are always parallel to the x-axis.
It is easy to check that the above quantities satisfy the charge and current
neutrality conditions.  
Note that the electron and ion Alfven speeds are defined by
$V_{A,e} = B_0 / \sqrt{n_0 m_e}$ and $V_{A,i} = B_0 / \sqrt{n_0 m_i}$,
respectively.


# Running

Build the `foot` target from the top-level build directory:
```
$ cmake --build build --target foot
```

Each scenario is in its own subdirectory containing `config.toml`.
To run, go to the scenario directory and execute:
```
$ cd build/pic/example/foot/<scenario>
$ export OMP_NUM_THREADS=2
$ mpiexec -n 8 ./main.out -e 86400 -t <duration> -c config.toml
```

Set the environment variable `PICNIX_DIR` to the path to the pic-nix root
directory. Then, run the quicklook script to examine the simulation results:
```
$ python quicklook.py data/profile.msgpack
```

## Example 1: buneman
Electrostatic ion-electron beam instability.
```
$ mpiexec -n 8 ./main.out -e 86400 -t 200 -c config.toml
```

## Example 2: alfven
Ion beam generating Alfven waves via the cyclotron resonance.
For details, see Hoshino & Terasawa (1985).
```
$ mpiexec -n 16 ./main.out -e 86400 -t 5000 -c config.toml
```

## Example 3: weibel
Perpendicular shock ($\theta = 90^\circ$) with filamentation-type instabilities.
```
$ mpiexec -n 16 ./main.out -e 86400 -t 5000 -c config.toml
```


# References
- Hoshino, M., & Terasawa, T. (1985). Numerical Study of the Upstream Wave
  Excitation Mechanism, 1. Nonlinear Phase Bunching of Beam Ions.
  *Journal of Geophysical Research*, *90*(A1), 57–64.
  https://doi.org/10.1029/JA090iA01p00057
- Matsukiyo, S., & Scholer, M. (2003). Modified two-stream instability in
  the foot of high Mach number quasi-perpendicular shocks.
  *Journal of Geophysical Research: Space Physics*, *108*(A12), 1459.
  https://doi.org/10.1029/2003JA010080
- Matsukiyo, S., & Scholer, M. (2006). On microinstabilities in the foot of
  high Mach number perpendicular shocks.
  *Journal of Geophysical Research: Space Physics*, *111*(6), 1–10.
  https://doi.org/10.1029/2005JA011409
- Amano, T., & Hoshino, M. (2009). Nonlinear evolution of Buneman instability
  and its implication for electron acceleration in high Mach number collisionless
  perpendicular shocks. *Physics of Plasmas*, *16*(10), 102901.
  https://doi.org/10.1063/1.3240336
- Muschietti, L., & Lembège, B. (2013). Microturbulence in the electron
  cyclotron frequency range at perpendicular supercritical shocks.
  *Journal of Geophysical Research: Space Physics*, *118*(5), 2267–2285.
  https://doi.org/10.1002/jgra.50224
