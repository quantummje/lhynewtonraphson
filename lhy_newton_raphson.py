import numpy as np
from numpy import linalg as la
from scipy import linalg
import matplotlib.pyplot as plt
import time
#
def findiff(mu0,n0,dbg,cpar,eps,p,Dx,tol):
    #
    x = np.arange(-Dx[0],Dx[0]+Dx[1],Dx[1])
    y = np.zeros([x.shape[0],1],dtype='float')
    #
    aout = []
    pout = []
    nout = []
    #
    run_flag = True
    #
    mu = mu0
    #
    c = 0
    #
    csmp = cpar[0]#1e3
    cmin = cpar[1]#1e4
    cmax = cpar[2]#1e5
    #
    for ii in xrange(0,x.shape[0]):
        y[ii] = np.sqrt(n0)*(np.tanh(np.sqrt(np.abs(mu))*(x[ii] - 0*10.)) + 0*1e-5*np.random.rand(1))
    #
    print(repr(y[0]) +' '+ repr(y[-1]))
    pout.append(y)
    aout.append(Dx[1]*np.trapz(y[0,0]**2 - y[:,0]**2))
    #
    dy = linalg.solve(J(Dx[1],y,p,mu), -f(Dx[1],y,p,mu))
    print(np.allclose(np.dot(J(Dx[1],y,p,mu), dy), -f(Dx[1],y,p,mu)))
    #
    while run_flag:
        #
        if c <= cmin:
            y = y + eps[0]*dy
        else:
            y = y + eps[1]*dy
        #
        c = c + 1
        #
        if divmod(c,csmp)[1] == 0:
            #
            if dbg == True:
                #
                print('Frob. norm: ' + repr(la.norm(dy,2)))
                print('mu: ' + repr(mu))
                print(repr(y[0]) +' '+ repr(y[-1]))
                print('Close check: ' + str(np.allclose(np.dot(J(Dx[1],y,p,mu), dy), -f(Dx[1],y,p,mu))))
                print('Atom number: ' + repr(Dx[1]*np.trapz(y[0,0]**2 - y[:,0]**2)))
                print(repr(c) + ' iterations completed')
                print('-------------------------------')
            #
            pout.append(y)
            nout.append(la.norm(dy,2))
            aout.append(Dx[1]*np.trapz(y[0,0]**2 - y[:,0]**2))
            #
            if la.norm(dy,2) <= tol and c >= cmin:
                #
                run_flag = False
                #
            if c >= cmax:
                #
                run_flag = False
                #
            #
        #
        dy = linalg.solve(J(Dx[1],y,p,mu), -f(Dx[1],y,p,mu))
        #
    #
    print('Frob. norm: ' + repr(la.norm(dy,2)))
    print('Atom number: ' + repr(Dx[1]*np.trapz(y[0,0]**2 - y[:,0]**2)))
    print('Energy: ' + repr(get_en(psi=y,p=p,dx=Dx[1])))
    print('Chemical potential: ' + repr(get_mu(psi=y,p=p,dx=Dx[1])))
    print(repr(c) + ' iterations completed')
    #
    return x, y, aout, pout, nout, mu
#
def f(Dx,y,p,mu):
    #
    n = y.shape[0]
    h = Dx
    #
    y1 = np.zeros([n,1])
    y1[0,0] = -0.5*(y[1,0] - y[0,0])/h**2 - p*pow(abs(1-p),-1)*np.sign(y[0,0])*y[0,0]**2 + pow(abs(1-p),-1)*y[0,0]**3 - mu*y[0,0]#y[0,0] + 1#np.sqrt(mu/g)
    #
    for ii in xrange(1,n-1):
        #
        y1[ii,0] = -0.5*(y[ii-1,0] - 2*y[ii,0] + y[ii+1,0])/h**2 - p*pow(abs(1-p),-1)*np.sign(y[ii,0])*y[ii,0]**2 + pow(abs(1-p),-1)*y[ii,0]**3 - mu*y[ii,0]
    #
    y1[n-1,0] = -0.5*(y[n-2,0] - y[n-1,0])/h**2 - p*pow(abs(1-p),-1)*np.sign(y[n-1,0])*y[n-1,0]**2 + pow(abs(1-p),-1)*y[n-1,0]**3 - mu*y[n-1,0]#y[n-1,0] - 1#np.sqrt(mu/g)
    #
    return y1
#
def J(Dx,y,p,mu):
    #
    n = y.shape[0]
    h = Dx
    #
    J1 = np.zeros([n,n])
    J1[0,0] = 0.5/h**2 - 2*p*pow(abs(1-p),-1)*np.sign(y[0,0])*y[0,0] + 3*pow(abs(1-p),-1)*y[0,0]**2 - mu
    J1[0,1] = -0.5/h**2
    #
    for ii in range(1,n-1):
        #
        J1[ii,ii+1] = -0.5/h**2
        J1[ii,ii] = 1/h**2 - 2*p*pow(abs(1-p),-1)*np.sign(y[ii,0])*y[ii,0] + 3*pow(abs(1-p),-1)*y[ii,0]**2 - mu
        J1[ii,ii-1] = -0.5/h**2
    #
    J1[n-1,n-2] = -0.5/h**2
    J1[n-1,n-1] = 0.5/h**2 - 2*p*pow(abs(1-p),-1)*np.sign(y[-1,0])*y[-1,0] + 3*pow(abs(1-p),-1)*y[-1,0]**2 - mu
    #
    return J1
#
def get_mu(psi,p,dx):
    #
    ke = 0
    vdw = 0
    mu = 0
    #
    for ii in range(1,psi.shape[0]-1):
        ke = ke + 0.5*dx*np.abs((psi[ii+1] - psi[ii-1])/(2*dx))**2
    #
    vdw = -p*pow(abs(1-p),-1)*dx*np.trapz(np.abs(psi[:,0])**3) + pow(abs(1-p),-1)*dx*np.trapz(np.abs(psi[:,0])**4)
    mu = (ke + vdw)/(dx*np.trapz(np.abs(psi[:,0])**2))
    #
    return mu
#
def get_en(psi,p,dx):
    #
    ke = 0
    ie = 0
    en = 0
    #
    for ii in range(1,len(psi)-1):
        ke = ke + dx*0.5*np.abs((psi[ii+1,0]-psi[ii-1,0])/(2.0*dx))**2
    #
    ie = 0.5*pow(1-p,-1)*dx*np.trapz(np.abs(psi[:,0])**4) - (2./3.)*p*pow(1-p,-1)*dx*np.trapz(np.abs(psi[:,0])**3)
    en = ke + ie
    en = en/(dx*np.trapz(np.abs(psi[:,0])**2))
    #
    return en
#

Dx = (20,0.1)
Dt = 1e-5
tol = 1e-6
n0 = pow(0.7,2)
p0 = (1. - pow(10,-5))*1.5*np.sqrt(n0);
mu0 = -p0*pow(abs(1-p0),-1)*pow(n0,0.5) + pow(abs(1-p0),-1)*n0;

#

cpar = (1e3,1e4,1e5)
dbg = True
eps = (2e-3,1.5e-4)

#

t0 = time.time()
x, psi, aout, pout, nout, muo = findiff(mu0=mu0,n0=n0,dbg=dbg,cpar=cpar,eps=eps,p=p0,Dx=Dx,tol=tol)
print('Time: ' + repr(time.time()-t0))
print('Norm: ' + repr(Dx[1]*np.trapz(psi[0,0]**2 - psi[:,0]**2)))

#

f1, ax = plt.subplots()
ax.plot(x,np.sqrt(n0)*np.tanh(np.sqrt(np.abs(mu0))*x),'--k',label='Zakharov-Shabat soliton ($\mu\simeq$|'+repr(np.abs(1e-2*np.round(mu0*100)))+'|)')
ax.plot(x[::4],psi[:,0][::4],'.-r',label='Numerical (NR) $\psi, p_0=$'+repr(p0))
ax.set_xlabel('$x$')
ax.set_ylabel('$\psi$')
ax.legend(bbox_to_anchor=(0, 1.15), loc='upper left', borderaxespad=0, ncol=2)
ax.grid()
ax.set_xlim([-Dx[0],Dx[0]])
f1.show()

#

f2 = plt.figure()
plt.subplot(121)
newp = np.array(pout)
plt.pcolormesh(newp.reshape(newp.shape[0],newp.shape[1]).T,cmap='seismic')
plt.subplot(122)
plt.semilogy(np.array(nout),'+')
f2.show()

#
