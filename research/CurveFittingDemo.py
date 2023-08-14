import matplotlib.pyplot as plt
import numpy as np

def traj(x,coeff):

    coeff = np.reshape(coeff,(4,1))

    num = len(x)
    x = np.reshape(x,(num,1))

    X = np.hstack((x**3, x**2, x, np.ones_like(x)))

    y = X@coeff

    return y

def path(x0,y0,xf,yf,traj_coeff):
    a = float(traj_coeff[0])
    b = float(traj_coeff[1])
    c = float(traj_coeff[2])
    d = float(traj_coeff[3])

    A = np.array([
        [a*x0**3, b*x0**2, c*x0, d],
        [3*a*x0**2, 2*b*x0, c, 0],
        [a*xf**3, b*xf**2, c*xf, d],
        [3*a*xf**2, 2*b*xf, c, 0]
    ])
    A = np.array([
        [x0**3, x0**2, x0, 1],
        [3*x0**2, 2*x0, 1, 0],
        [xf**3, xf**2, xf, 1],
        [3*xf**2, 2*xf, 1, 0]
    ])

    b = np.array([
        [float(y0)],
        [0.],
        [float(yf)],
        [float(3*a*xf**2 + 2*b*xf + c)]
    ])

    coeff = np.linalg.inv(A)@b

    x = np.linspace(x0, xf, 50)
    y = traj(x,coeff)

    return x,y,coeff

def curvature(pos,coeff):
    yp =  np.array([
        [3*coeff[0]],
        [2*coeff[1]],
        [1*coeff[2]],
        [0*coeff[3]]
    ])
    ypp = yp*np.array([
        [2],
        [1],
        [0],
        [0]
    ])
    curve = 1
    return curve

def coefficients():
    coeff = np.zeros((4,1))
    coeff[0] = np.random.normal(loc=0,scale=0.01,size=1)
    coeff[1] = np.random.normal(loc=0,scale=0.01,size=1)
    coeff[2] = np.random.normal(loc=0,scale=1.0,size=1)
    coeff[3] = np.random.normal(loc=0,scale=1.0,size=1)

    return coeff

if __name__ == '__main__':


    for i in range(20):
        coeff = coefficients()


        x1 = np.linspace(0,5,100)
        print(x1)

        y1 = traj(x1,coeff)

        pos = np.random.normal(scale=0.1)
        idx = 20
        x2,y2,coeff = path(x0=0.,y0=pos, xf=x1[idx],yf=y1[idx],traj_coeff=coeff)

        plt.figure(1,figsize=(8,8))
        plt.plot(y1-y2[0],x1,'b-')
        plt.plot(y2-y2[0],x2,'r--')
        plt.axis('equal')
        plt.pause(0.8)
        plt.close()