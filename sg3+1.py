import autograd.numpy as np
from autograd import jacobian,grad, holomorphic_grad
import autograd.numpy.random as npr
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

nx = 10
ny = 10
nz = 10
nt = 10

dx = 1. / nx
dy = 1. / ny
dz = 1. / nz
dt = 1. / nt

x_space = np.linspace(0, 1, nx)
y_space = np.linspace(0, 1, ny)
z_space = np.linspace(0, 1, nz)
t_space = np.linspace(0, 1, nt)


def analytic_solution(x):
    return 4*np.arctan(np.exp(0.707*x[0]+0.5*(x[1]+1.54*x[2]+1.175*x[3]))) 

def f(x):
    return 0.

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def neural_network(W, x):
    a1 = sigmoid(np.dot(x, W[0]))
    return np.dot(a1, W[1])


def neural_network_x(x):
    a1 = sigmoid(np.dot(x, W[0]))
    return np.dot(a1, W[1])

def A(x):
    return (x[0]*x[1]*x[2]+x[0]*x[1]*(1-x[2])+x[0]*(1-x[1])*(1-x[2])+x[0]*(1-x[1])*x[2]+(1-x[0])*x[1]*x[2]+(1-x[0])*x[1]*(1-x[2])+(1-x[0])*(1-x[1])*(1-x[2])+(1-x[0])*(1-x[1])*x[2])*4*np.arctan(np.exp(0.707*x[0]+0.5*(x[1]+1.54*x[2])))+(1-x[0])*4*np.arctan(np.exp(0.5*(x[1]+1.54*x[2]+1.175*x[3])))*(x[1]*x[2]+x[1]*(1-x[2])+(1-x[1])*x[2]+(1-x[1])*(1-x[2]))+(x[0])*4*np.arctan(np.exp(0.707+0.5*(x[1]+1.54*x[2]+1.175*x[3])))*(x[1]*x[2]+x[1]*(1-x[2])+(1-x[1])*x[2]+(1-x[1])*(1-x[2]))+(1-x[1])*4*np.arctan(np.exp(0.707*x[0]+0.5*(1.54*x[2]+1.175*x[3])))*(x[0]*x[2]+x[0]*(1-x[2])+(1-x[0])*x[2]+(1-x[0])*(1-x[2]))+x[1]*4*np.arctan(np.exp(0.707*x[0]+0.5*(1+1.54*x[2]+1.175*x[3])))*(x[0]*x[2]+x[0]*(1-x[2])+(1-x[0])*x[2]+(1-x[0])*(1-x[2]))+(1-x[2])*4*np.arctan(np.exp(0.707*x[0]+0.5*(x[1]+1.175*x[3])))*(x[0]*x[1]+x[0]*(1-x[1])+(1-x[0])*x[1]+(1-x[0])*(1-x[1]))+x[2]*4*np.arctan(np.exp(0.707*x[0]+0.5*(x[1]+1.54+1.175*x[3])))*(x[0]*x[1]+x[0]*(1-x[1])+(1-x[0])*x[1]+(1-x[0])*(1-x[1]))+x[3]*2*1.175*(np.exp(0.707*x[0]+0.5*(x[1]+1.54*x[2])))/(1+(np.exp(1.41*x[0]+x[1]+1.54*x[2])))*(x[0]*x[1]*x[2]+x[0]*x[1]*(1-x[2])+x[0]*(1-x[1])*x[2]+x[0]*(1-x[1])*(1-x[2])+(1-x[0])*x[1]*x[2]+(1-x[0])*x[1]*(1-x[2])+(1-x[0])*(1-x[1])*x[2]+(1-x[0])*(1-x[1])*(1-x[2]))


def psy_trial(x, net_out):
    return A(x) + x[0] * (1 - x[0]) * x[1] * (1 - x[1]) *x[2] * (1 - x[2])*x[3]**2* net_out


def Lossfunction(W, x, y, z, t):
    loss_sum = 0
    
    for xi in x:
        for yi in y:
            for zi in z:
                for ti in t:
                    input_point = np.array([xi, yi, zi, ti])
            
                    net_out = neural_network(W, input_point)[0]
  
                    net_out_jacobian = jacobian(neural_network_x)(input_point)
                    net_out_hessian = jacobian(jacobian(neural_network_x))(input_point)
            
                    psy_t = psy_trial(input_point, net_out)
                    psy_t_jacobian = jacobian(psy_trial)(input_point, net_out)
                    psy_t_hessian = jacobian(jacobian(psy_trial))(input_point, net_out)

                    gradient_of_trial_d2x = psy_t_hessian[0][0]
                    gradient_of_trial_d2y = psy_t_hessian[1][1]
                    gradient_of_trial_d2z = psy_t_hessian[2][2]
                    gradient_of_trial_d2t = psy_t_hessian[3][3]

                    func = np.sin(psy_trial(input_point, net_out)) 

                    err_sqr = ((gradient_of_trial_d2t - gradient_of_trial_d2x - gradient_of_trial_d2y - gradient_of_trial_d2z) - func)**2
                    loss_sum += err_sqr
    return loss_sum

  
W = [npr.randn(4, 10), npr.randn(10, 1)]
lmb = 0.001

res = np.zeros((nx, ny*nz*nt))
res_analytic = np.zeros((nx, ny*nz*nt))

for i in range(10):
    lossgrad =  grad(Lossfunction)(W, x_space, y_space, z_space, t_space)

    W[0] = W[0] - lmb * lossgrad[0]
    W[1] = W[1] - lmb * lossgrad[1]


for i, x in enumerate(x_space):
    for j, y in enumerate(y_space):
        for s, z in enumerate(z_space):
            for d, t in enumerate(t_space):
                for m=0:ny*nz*nt:
                    res_analytic[i][m] = analytic_solution([x, y, z, t])
        
for i, x in enumerate(x_space):
    for j, y in enumerate(y_space):
        for s, z in enumerate(z_space):
            for d, t in enumerate(t_space):
                for m=0:ny*nz*nt:
    	            net_outt = neural_network(W, [x, y, z, t])[0]
    	            res[i][m] = psy_trial([x, y, z, t], net_outt)


print (res)
print("start")
print (res_analytic)
        
