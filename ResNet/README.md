ResNet

Paper: Deep Residual Learning for Image Recognition - Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun

Evidence shows increased NN depth is of crucial importance to developing accurate and models that deal with more complex problems. However some issues with increasing NN depth:
 - Vanishing/Exploding Gradient Problem was an obstacle but has been largely addressed by normalization layers
 - Degradation Problem: increasing network depth quickly saturates the accuracy which then degrades quickly after. What we see is that just
adding more layers leads to higher training error. This degradation is not caused by overfitting, but is instead an optimization problem.


Papers proposed solution to the degradation problem:
A Deep Residual Learning framework. Instead of hoping each few stacked layers fit the desired underlying mapping G(x), we let the stacked nonlinear layers fit
the mapping F(x) = G(x) - x, where F(x) is the underlying mapping for the nonlinear stack, and x is the identity (linear mapping). Then to optimize G(x) we can optimize
G(x) = F(x) + x where x acts like a residual. Much easier to minimize the residual than to try and fit the identity mapping.

This residual framework can be realized by "shortcut" connections - Fig2 in Paper

ResNet 50 architecture that I implemented: http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006
