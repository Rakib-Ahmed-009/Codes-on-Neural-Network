import autograd.numpy as np
from autograd import grad, elementwise_grad
import autograd.numpy.random as npr
from autograd.misc.optimizers import adam

def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    
    return [(rs.randn(insize, outsize) * scale,   
             rs.randn(outsize) * scale)           
            for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]


def swish(x):
    
    return x / (1.0 + np.exp(-x))


def Ca(params, inputs):
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = swish(outputs)    
    return outputs

    

params = init_random_params(0.1, layer_sizes=[1, 8, 10, 8, 1])


dCadt = elementwise_grad(Ca, 1)
ddCadt = elementwise_grad(dCadt, 1)

k = 1
Ca0 = 1
t = np.linspace(0, 1).reshape((-1, 1))


def objective(params, step):
    
    zeq = ddCadt(params, t) - (-k * Ca(params, t))
    ic1 = Ca(params, 1) - Ca0
    ic2 = Ca(params,0)
    
    return np.mean(zeq**2) + (ic1**2) + (ic2**2)

def callback(params, step, g):
    if step % 1000 == 0:
        print("Iteration {0:3d} objective {1}".format(step,
                                                      objective(params, step)))

params = adam(grad(objective), params,
              step_size=0.001, num_iters=500, callback=callback) 


tfit = np.linspace(0, 1).reshape(-1, 1)
import matplotlib.pyplot as plt
plt.plot(tfit, Ca(params, tfit), label='NN soln')
plt.plot(tfit, (np.sin(tfit))/(np.sin(1)), 'r--', label='Analytical soln')
plt.legend()
plt.xlabel('Position')
plt.ylabel('Displacement')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()
plt.savefig('nn-ode.png')
