# Hybrid Optional Elliptic Backend  

The Hybrid beam port currently uses the faithful legacy SSOR2 Ohm solver.  
An elliptic/PETSc-backed Ohm solve remains intentionally deferred.  

Future work should preserve the same discrete Ohm system before changing solver backend behavior:  

- Add a Hybrid-owned accessor over the existing Ohm coefficient, RHS, and electric-field solution arrays.  
- Keep PETSc opt-in with `PICNIX_ENABLE_PETSC=ON`; default builds must not search for or link PETSc.  
- Keep the legacy SSOR2 backend selectable and covered by tests.  
- Compare SSOR2 and elliptic backends on identical manufactured periodic systems.  
- Compare short Hybrid beam runs with both backends using documented solver tolerances.  
- Rebuild any backend mapping after restart and DLB.  

Stop and redesign if the elliptic backend would require a `pic/` dependency, change the initial discrete operator, or make PETSc mandatory.  
