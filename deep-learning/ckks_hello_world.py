import sys
sys.path.append("../") #syft library path

import syft as sy
import torch as th
import syft.frameworks.tenseal as ts

import time

DEFAULT_CKKS_N = 8192*4
DEFAULT_CKKS_COEFF_MOD = [60, 60, 60, 60, 60, 60, 60, 60]
DEFAULT_CKKS_SCALE = 2 ** 40
DIMS = 128


# hook PyTorch to add extra functionalities like the ability to encrypt torch tensors
hook = sy.TorchHook(th)

# Generate CKKS public and secret keys
public_keys, secret_key = ts.generate_ckks_keys(poly_modulus_degree=DEFAULT_CKKS_N, \
                                                coeff_mod_bit_sizes=DEFAULT_CKKS_COEFF_MOD)

#matrix = th.tensor([[10.5, 73, 65.2], [13.33, 22, 81]])
matrix = th.rand(DIMS, DIMS)
matrix_encrypted = matrix.encrypt("ckks", public_key=public_keys)

# to use for plain evaluations
t_eval = th.tensor([[1, 2.5, 4], [13, 7, 16]])
t_eval = th.rand(DIMS, DIMS)
# to use for encrypted evaluations
t_encrypted = t_eval.encrypt("ckks", public_key=public_keys)

print("encrypted tensor + plain tensor")
t1 = time.time()
result = matrix_encrypted + t_eval
print("Elapsed time: %s"%(time.time()-t1))
# result is an encrypted tensor
result = result.decrypt(secret_key=secret_key, protocol="ckks")

print("encrypted tensor + encrypted tensor")
t1 = time.time()
result = matrix_encrypted + t_encrypted
print("Elapsed time: %s"%(time.time()-t1))
# result is an encrypted tensor
result = result.decrypt(secret_key=secret_key, protocol="ckks")

print("encrypted tensor - plain tensor")
t1 = time.time()
result = matrix_encrypted - t_eval
print("Elapsed time: %s"%(time.time()-t1))
# result is an encrypted tensor
result = result.decrypt(secret_key=secret_key, protocol="ckks")

print("encrypted tensor - encrypted tensor")
t1 = time.time()
result = matrix_encrypted - t_encrypted
print("Elapsed time: %s"%(time.time()-t1))
# result is an encrypted tensor
result = result.decrypt(secret_key=secret_key, protocol="ckks")

print("encrypted tensor * plain tensor")
t1 = time.time()
result = matrix_encrypted * t_eval
print("Elapsed time: %s"%(time.time()-t1))
# result is an encrypted tensor
result = result.decrypt(secret_key=secret_key, protocol="ckks")

print("encrypted tensor * encrypted tensor")
t1 = time.time()
result = matrix_encrypted * t_encrypted
print("Elapsed time: %s"%(time.time()-t1))
# result is an encrypted tensor
result = result.decrypt(secret_key=secret_key, protocol="ckks")

