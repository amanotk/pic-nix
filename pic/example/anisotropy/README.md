# Temperature Anisotropy Driven Instability

This setup models the temperature anisotropy driven instability in  
a magnetized plasma with periodic boundary conditions in all three directions.  
The system consists of bi-Maxwellian electrons and ions.

The electron whistler instability is excited when the electron perpendicular
temperature exceeds the parallel temperature beyond a threshold:
$T_{\perp}/T_{\parallel} > 1 + 1/\beta_{\parallel}$ (Gary 1993).

## Physical Parameters

The following parameters should be defined in the configuration file:

* `cc` : speed of light $c$
* `wp` : electron plasma frequency $\omega_{pe}$
* `mime` : ion-to-electron mass ratio $m_i/m_e$
* `theta` : polar angle of the ambient magnetic field with respect to the
  x-axis $\theta$
* `phi` : azimuthal angle of the ambient magnetic field with respect to the
  x-axis $\phi$
* `sigma` : electron cyclotron-to-plasma frequency squared
  $\sigma = \Omega_{ce}^2/\omega_{pe}^2$
* `betae_para` : electron parallel plasma beta $\beta_{\parallel,e}$
* `betae_perp` : electron perpendicular plasma beta $\beta_{\perp,e}$
* `betai_para` : ion parallel plasma beta $\beta_{\parallel,i}$
* `betai_perp` : ion perpendicular plasma beta $\beta_{\perp,i}$
* `nppc` : number of particles per cell per species
* `delt` : time step $\Delta t$ in units of $\omega_{pe}^{-1}$
* `delh` : grid spacing $\Delta h$ in units of $c/\omega_{pe}$

The electron charge-to-mass ratio is assumed to be unity $|e|/m_e = 1$,
which gives the ambient magnetic field $B_0 = c \sqrt{\sigma}$.

## Bi-Maxwellian Distribution

The bi-Maxwellian distribution function is given by
```math
f(v_\parallel, v_\perp) =
\frac{1}{\left( 2 \pi \right)^{3/2} v_{th,\parallel} v_{th,\perp}^2}
\exp
\left[
    - \frac{v_\parallel^2}{2 v_{th,\parallel}^2}
    - \frac{v_\perp^2}{2 v_{th,\perp}^2}
\right]
```

The particle initialization draws velocities from three independent Gaussian
distributions:
* $v_\parallel \sim \mathcal{N}(0, v_{th,\parallel}^2)$
* $v_{\perp,1} \sim \mathcal{N}(0, v_{th,\perp}^2)$
* $v_{\perp,2} \sim \mathcal{N}(0, v_{th,\perp}^2)$

The thermal speeds are related to the plasma beta via
```math
v_{th,\parallel} = v_A \sqrt{\frac{\beta_{\parallel}}{2}}, \qquad
v_{th,\perp} = v_A \sqrt{\frac{\beta_{\perp}}{2}},
```
where $v_A = c \sqrt{\sigma}$ is the electron Alfvén speed.

## References

* Gary, S. P. (1993). *Theory of Space Plasma Microinstabilities*.
  Cambridge University Press.
* Gary, S. P., & Wang, J. (1996). Whistler instability: Electron temperature
  anisotropy in the solar wind. *Journal of Geophysical Research*,
  *101*(A5), 10749–10758. https://doi.org/10.1029/96JA00354
