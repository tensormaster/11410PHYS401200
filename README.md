# 11410PHYS401200

https://github.com/pcchen/11410PHYS401200/

pcchen@phys.nthu.edu.tw

## Links
* https://exercism.org/
* https://sci-tai.vm.nthu.edu.tw/signin
* https://github.com/smorita/TRG_Ising_2D
* https://smorita.github.io/TN_animation/
* https://itensor.org/docs.cgi?page=book/trg
* https://github.com/Cytnx-dev/Cytnx
* https://www.scipost.org/SciPostPhysCodeb.53
* 
## 2D Ising Model

Z(2x2)(T=1, h=0) = 5973.916645008712

## Assignment-1
For 1D Ising model
* Find/Derive analytical expression for transfer matrix at any temperture T and external field h.
* Find/Derive analytical expression for free energy per site for L-sites system.
* Code: Use exact summation to evaluate partition function, and free energy per site for L-sites system.
* Code: Use transfer matrix to evaluate partition function, and free energy per site for L-sites system.
* Code: Plot free energy per site v.s. T, with different L.
* Code: Plot specific heat per site v.s. T, with different L.

For 2D Ising model
* Code: Use exact summation to evaluate partition function, and free energy per site for Lx * Ly system.
* Code: Use transfer matrix to evaluate partition function, and free energy per site for Lx * Ly system.
* Code: Plot free energy per site v.s. T, with (Lx,Ly)=(2,2), (3,2), (2,3,), (3,3), (3,4), (4,3), (4,4), etc
* Code: Plot specific heat per site v.s. T, with different (Lx,Ly)=(2,2), (3,2), (2,3,), (3,3), (3,4), (4,3), (4,4), etc

## Assignment-2
For 2D Ising model
* Construct rank-2 tensor (=matrix) M
* Construct rank-4 tensor T from M
* When temp=1=J, element of T includes <br>
  [[4.76220e+00 0.00000e+00 0.00000e+00 3.62686e+00 ] <br>
 [0.00000e+00 3.62686e+00 3.62686e+00 0.00000e+00 ] <br>
 [0.00000e+00 3.62686e+00 3.62686e+00 0.00000e+00 ] <br>
 [3.62686e+00 0.00000e+00 0.00000e+00 2.76220e+00 ]] <br>
* Contract four T tensors to get Z_2x2
* Use numpy.linalg.svd and numpy.linalg.eigh to find M-matrix from W-matrix.

## Assignment-3
* Use `list` and `np.array` to create vector=rank-1 tensor and matrix=rank-2 tensor.
* Perform vector dot and matrix multiplication, study/plot time v.s. dimension.
* Try numpy.tensordot and numpy.einsum.

## Assignment-4
* Perform SVD on nxn matrix and truncate it, see how error increases as you truncate more singular values.

