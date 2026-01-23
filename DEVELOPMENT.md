# Development  
  
This document describes the local developer workflow.  
It focuses on Catch2 v3 setup for unit testing.  
  
## Catch2 v3 Setup  
Prefer an external Catch2 v3 install and point CMake at its config  
file.  
Use the helper script to install Catch2 v3 into a custom prefix.  
  
```sh
script/install_catch2v3.sh "$HOME/usr"
```
  
Then configure tests with the explicit config path:  
  
```sh
cmake -S . -B build \
  -DBUILD_TESTING=ON \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DPICNIX_CATCH2_CONFIG="$HOME/usr/lib/cmake/Catch2/Catch2Config.cmake"
```
  
## Smoke Test Golden Data  
`pic/unittest/test_pic_chunk.cpp` writes smoke test golden data to  
`pic/unittest/testdata/pic_chunk_smoke_*_{field,particle}.msgpack`.  
Each msgpack file stores a map keyed by option tags (e.g.,  
`vector_o3_Vay_WT`) so multiple option sweeps live in one file.  
Regenerate the data by setting `PICNIX_UPDATE_GOLDEN=1` when running  
`test_pic_chunk`.  
  
## Git Hooks  
Install the local pre-commit hook (clang-format on staged C/C++) after  
cloning:  
  
```sh
script/git-hooks/install.sh
```
  
## CI Notes  
The GitHub Actions workflow installs Catch2 v3 externally and sets  
`PICNIX_CATCH2_CONFIG` for the test builds.  
