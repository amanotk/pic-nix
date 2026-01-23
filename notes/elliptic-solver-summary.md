# Elliptic Solver Edit Summary  
- Context: `test_pic_poisson_np2` failed because periodic Laplacian assembly overwrote overlapping neighbor contributions on 2-cell grids.  
- Change: in `elliptic/petsc_poisson.cpp` switched all `MatSetValuesStencil` calls in 1D/2D/3D assembly from `INSERT_VALUES` to `ADD_VALUES` so shared entries accumulate properly.  
- Effect: periodic Poisson matrix now matches intended stencil and computed potential aligns with reference.  
- Tests: `ctest --test-dir build -R test_pic_poisson_np2 --output-on-failure` (pass).  
