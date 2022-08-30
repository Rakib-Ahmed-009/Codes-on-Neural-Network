import autograd.numpy as np
from autograd import hessian,grad
import autograd.numpy.random as npr
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def u(t):
    return 4*np.arctan(np.exp(-1.71*t))
def v(t):
    return 4*np.arctan(np.exp(2-1.71*t))
def f(x):
    return 4*np.arctan(np.exp(2*x))

def g(x):
    return -4*1.71*(np.exp(2*x))/(1+np.exp(4*x))

def h1(point):
    x,t = point
    return (1-x)*u(t)+x*v(t)+f(x)-3.1416*(1-x)-4*x*np.arctan(np.exp(2))+t*g(x)+3.42*t*(1-x)+6.48*t*x*(np.exp(2))/(1+np.exp(4))

def g_trial(point,P):
    x,t = point
    return h1(point) + x*(1-x)*t**2*nn(P,point)


def cost_function(P, x, t):
    cost_sum = 0

    g_t_hessian_func = hessian(g_trial)

    for x_ in x:
        for t_ in t:
            point = np.array([x_,t_])

            g_t_hessian = g_t_hessian_func(point,P)

            g_t_d2x = g_t_hessian[0][0]
            g_t_d2t = g_t_hessian[1][1]
            sinterm = np.sin(g_trial(point,P))

            err_sqr = ( (g_t_d2t - g_t_d2x - sinterm) )**2
            cost_sum += err_sqr

    return cost_sum / (np.size(t) * np.size(x))


def sigmoid(z):
    return 1/(1 + np.exp(-z))

def nn(deep_params, x):
    
    num_coordinates = np.size(x,0)
    x = x.reshape(num_coordinates,-1)

    num_points = np.size(x,1)

    
    N_hidden = np.size(deep_params) - 1 

    
    x_input = x
    x_prev = x_input

    

    for l in range(N_hidden):
        
        w_hidden = deep_params[l]

        
        x_prev = np.concatenate((np.ones((1,num_points)), x_prev ), axis = 0)

        z_hidden = np.matmul(w_hidden, x_prev)
        x_hidden = sigmoid(z_hidden)

        
        x_prev = x_hidden

    
    w_output = deep_params[-1]

    
    x_prev = np.concatenate((np.ones((1,num_points)), x_prev), axis = 0)

    z_output = np.matmul(w_output, x_prev)
    x_output = z_output

    return x_output[0][0]


def g_analytic(point):
    x,t = point
    return 4*np.arctan(np.exp(2*(x-0.866*t)))

def solve_pde(x,t, num_neurons, num_iter, lmb):
    
    N_hidden = np.size(num_neurons)

    
    P = [None]*(N_hidden + 1) 
    P[0] = npr.randn(num_neurons[0], 2 + 1 ) 
    for l in range(1,N_hidden):
        P[l] = npr.randn(num_neurons[l], num_neurons[l-1] + 1) 

    
    P[-1] = npr.randn(1, num_neurons[-1] + 1 ) 

    print('Initial cost: ',cost_function(P, x, t))

    cost_function_grad = grad(cost_function,0)

    
    for i in range(num_iter):
        cost_grad =  cost_function_grad(P, x , t)

        for l in range(N_hidden+1):
            P[l] = P[l] - lmb * cost_grad[l]


    print('Final cost: ',cost_function(P, x, t))

    return P

if __name__ == '__main__':
    
    npr.seed(15)

    
    Nx = 10; Nt = 10
    x = np.linspace(0,1, Nx)
    t = np.linspace(0,1,Nt)

    
    num_hidden_neurons = [50,20]
    num_iter = 1000
    lmb = 0.001

    P = solve_pde(x,t, num_hidden_neurons, num_iter, lmb)

    
    res = np.zeros((Nx, Nt))
    res_analytical = np.zeros((Nx, Nt))
    for i,x_ in enumerate(x):
        for j, t_ in enumerate(t):
            point = np.array([x_, t_])
            res[i,j] = g_trial(point,P)

            res_analytical[i,j] = g_analytic(point)

    diff = np.abs((res**2) - (res_analytical**2))
    

    

    T,X = np.meshgrid(t,x)

    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    ax.set_title('Solution from the deep neural network w/ %d layer'%len(num_hidden_neurons))
    s = ax.plot_surface(T,X,res,linewidth=0,antialiased=False,cmap=cm.viridis)
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Position $x$');


    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    ax.set_title('Analytical solution')
    s = ax.plot_surface(T,X,res_analytical,linewidth=0,antialiased=False,cmap=cm.viridis)
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Position $x$');


    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    ax.set_title('Difference of $L^2$ Norms')
    s = ax.plot_surface(T,X,diff,linewidth=0,antialiased=False,cmap=cm.viridis)
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Position $x$');

    
    indx1 = 0
    indx2 = int(Nt/2)
    indx3 = Nt-1

    t1 = t[indx1]
    t2 = t[indx2]

    res1 = res[:,indx1]
    res2 = res[:,indx2]

    
    res_analytical1 = res_analytical[:,indx1]
    res_analytical2 = res_analytical[:,indx2]

    
    

    plt.figure(figsize=(10,10))
    plt.title("Computed solutions at time = %g"%t1)
    plt.plot(x, res1)
    plt.plot(x,res_analytical1)
    plt.legend(['nn','analytic'])
    plt.xlabel('Time $t$')
    plt.ylabel('Solution $\phi$');
    

    plt.figure(figsize=(10,10))
    plt.title("Computed solutions at time = %g"%t2)
    plt.plot(x, res2)
    plt.plot(x,res_analytical2)
    plt.legend(['nn','analytic'])
    plt.xlabel('Time $t$')
    plt.ylabel('Solution $\phi$');


    plt.figure(figsize=(10,10))
    plt.title("Computed Difference of $L^2$ Norms at time = %g"%t1)
    plt.plot(x, (res_analytical1**2)-(res1**2))
    plt.legend(['Difference at time t=0'])
    plt.xlabel('Time $t$')
    plt.ylabel('Difference of $L^2$ Norms ');

    plt.figure(figsize=(10,10))
    plt.title("Computed Difference of $L^2$ Norms at time = %g"%t2)
    plt.plot(x, (res_analytical2**2)-(res2**2))
    plt.legend(['Difference at t=0.55556'])
    plt.xlabel('Time $t$')
    plt.ylabel('Difference of $L^2$ Norms ');

    plt.show()
