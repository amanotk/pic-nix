Here is a human-written summary of the reoarganization plan for the hybrid module.

This is obviously not a definitive plan and is subject to change as we go along. The goal is to make the hybrid module more maintainable and easier to understand in the long run. Whenever LLM finds any objections or better ideas, we should discuss in detail and incorporate them into the plan.

# General
We expose nix::typedefs in the hybrid module by doing
```cpp
using namespace nix;
```
as we do in pic module. Accordingly, we will remove the `nix::` prefix from all the typedefs (such as `float64`)

Currently we take care of the background magnetic field in the hybrid module. At this moment, this is unnecessary complication and we will remove it.

# File Specific Notes
## hybrid_application.cpp
The push() method is huge and contains a lot of logic that is extremely difficult to follow.
This definitely needs to be broken up into smaller methods and refactored to improve readability and maintainability. You can refer to pic_application.cpp where the push() method is much cleaner.

## hybrid_chunk.hpp
I suspect many of the internal data members may be moved to somewhere else. They are used as working storage and specific to specific schemes used, which means that each scheme may have its own data members. This is clearly coupled with refactoring of engine module.

## hybrid_chunk.cpp
I suspect solver methods should be placed here as member functions of this class. This is the pattern used in pic_chunk.cpp. Each solver or scheme handler may be passed to them as an argument.

## engine module
We should not introduce structures such as FieldState or VectorState. Have a look at the original hybrid module in which these structures were not used and simply pointers to the data were passed around. This is a much cleaner approach and avoids unnecessary complexity.

## engine/pcc2.hpp
I am not sure why we need this level of abstraction, which to me seems unnecessary.

## engine/fluid.hpp
This provides indepenent functions used in fluid solver and may better be renamed to something else, such as fluid_utils.hpp or fulid_primitives.hpp (which may be a bit confusing potentially).

## engine/field.hpp and engine/mc2.hpp
They seem to provide the core solvers for the field/moment equations. Ideally, we should make them into one or maybe two handler classes that contain working storage necessary for themselves and solve the equations.

## engine/particle.hpp
We will eventually use the boris pusher provided in nix and do some refactoring on checing the cfl condition. This is of lower priority and can be done later.

## engine/interpolation.hpp
The particle shape function and interpolation may be removed as they are already implemented in nix module. However, this should be done carefully and is thus of lower priority.

## engine/phasespeed.hpp
Possibly remove structure definitions that may be unnecessary.

## engine/moment.hpp
Perhaps we may harmonize the implementation with the similar implementation in pic module.

## engine/ohm_*.hpp and engine/ssor2.hpp
We will eventually migrate to PETSc-based solvers, so we will not do any major refactoring unless it is necessary. One possible exception is that some working storage in hybrid_chunk.hpp may be moved to the solver classes.
