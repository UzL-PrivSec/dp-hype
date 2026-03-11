from autodp.mechanism_zoo import ExactGaussianMechanism
from autodp.transformer_zoo import Composition


## eps=0.1
# noise_scale = 103.0
## eps=0.25
# noise_scale = 46.0
## eps=0.5
# noise_scale = 24.0
## eps=1.0
# noise_scale = 12.5
## eps=3.0
# noise_scale = 4.7
## eps=inf
# noise_scale = 0.0

delta = 10**-5
num_comps = 1
k = 5
sens = (2 * k) ** 0.5

sigma = 12.5

mech = ExactGaussianMechanism(sigma / sens)

comp = Composition()
mech_compose = comp.compose([mech], [num_comps])

eps = mech_compose.get_approxDP(delta)

print(eps)
