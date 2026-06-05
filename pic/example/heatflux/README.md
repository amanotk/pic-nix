# Electron Heat Flux Instability

This setup models the electron beam instability in a magnetized plasma with a three-component homogeneous plasma with the periodic boundary condition in all three directions. The system consists of the beam electrons, the core (upstream) electrons, and the background ions. The beam electrons drift along the ambient magnetic field, and thus carry a heat flux. The core electrons drift in the opposite direction to maintain current neutrality.

The core electrons and ions are represented by the isotropic Maxwellian distribution, while the beam electrons are represented by a Maxwellian ring-beam distribution. The simulation frame corresponds to the rest frame of ions.

## Physical Parameters

The following parameters should be defined in the configuration file:
- `cc` : speed of light $c$
- `wp` : electron plasma frequency $\omega_{pe}$
- `mime` : ion-to-electron mass ratio $m_i/m_e$
- `theta` : polar angle of the ambient magnetic field with respect to the x-axis $\theta$
- `phi` : azimuthal angle of the ambient magnetic field with respect to the x-axis $\phi$
- `sigma` : electron cyclotron-to-plasma frequency squared $\sigma = \Omega_{ce}^2/\omega_{pe}^2$
- `vti` : ion thermal speed normalized to the speed of light $v_{th,i}/c$
- `vte` : core electron thermal speed normalized to the speed of light $v_{th,e}/c$
- `nb` : beam electron density fraction normalized to the total electron density $n_b/n_e$
- `vdb_para` : parallel beam electron drift velocity $V_{\parallel}/c$
- `vdb_perp` : perpendicular beam electron drift (or ring) velocity $V_{\perp}/c$
- `vtb_para` : beam electron thermal speed parallel to the magnetic field $v_{tb,\parallel}/c$
- `vtb_perp` : beam electron thermal speed perpendicular to the magnetic field $v_{tb,\perp}/c$
- `nppc` : number of particles per cell per species
- `delt` : time step $\Delta t$ in units of $\omega_{pe}^{-1}$
- `delh` : grid spacing $\Delta h$ in units of $c/\omega_{pe}$

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

The core electron drift velocity is determined by the current neutrality condition:
```math
v_{dc,\parallel} = -\frac{n_b}{1 - n_b} v_{db,\parallel}
```
This ensures that the total current density vanishes in the simulation frame.

The Maxwellian ring-beam distribution is defined as (see, e.g., Umeda et al. 2012)
```math
f(v_\parallel, v_\perp) =
\frac{A}{\left( 2 \pi \right)^{3/2} v_{th,\parallel} v_{th,\perp}^2}
\exp
\left[
    - \frac{(v_\parallel - V_{\parallel})^2}{2 v_{th,\parallel}^2}
    - \frac{(v_\perp - V_{\perp})^2}{2v_{th,\perp}^2}
\right]
```
where the constant $A$ is given by
```math
\frac{1}{A} =
\exp\left(-\frac{V_{\perp}^2}{2 v_{th,\perp}^2}\right) +
\sqrt{\frac{\pi}{2}}
\frac{V_{\perp}}{v_{th,\perp}}
 {\rm erfc}\left(-\frac{V_{\perp}}{\sqrt{2} v_{th,\perp}}\right).
```
It is readily seen that the distribution reduces to the standard bi-Maxwellian distribution when the ring velocity $V_{\perp}$ is zero.

## Scenarios

The default configuration file `config.toml` provides an example for a bi-Maxwellian electron beam that is unstable to the oblique whistler heat flux instability (see, e.g., Micera et al. 2020). The suggested run time is $\omega_{pe} t = 1000$.

## References

- Umeda, T., Matsukiyo, S., Amano, T., & Miyoshi, Y. (2012). A numerical electromagnetic linear dispersion relation for Maxwellian ring-beam velocity distributions. *Physics of Plasmas*, *19*(7), 072107.
  https://doi.org/10.1063/1.4736848
- Micera, A., Zhukov, A. N., López, R. A., Innocenti, M. E., Lazar, M., Boella, E., & Lapenta, G. (2020). Particle-in-cell Simulation of Whistler Heat-flux Instabilities in the Solar Wind: Heat-flux Regulation and Electron Halo Formation. *The Astrophysical Journal Letters*, *903*(1), L23.
  https://doi.org/10.3847/2041-8213/abc0e8
