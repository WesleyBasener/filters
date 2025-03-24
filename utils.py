import jax.numpy as np
from jax.numpy import cos, sin, atan2, asin

def cart_to_rbe(cart:np.array, covar=None):
    
    r = np.linalg.norm(cart)
    b = atan2(cart[1], cart[0])
    e = asin(cart[2]/r)

    return np.array([r,b,e])
    

def rbe_to_cart(rbe:np.array, covar=None):

    x = rbe[0]*cos(rbe[2])*cos(rbe[1])
    y = rbe[0]*cos(rbe[2])*sin(rbe[1])
    z = rbe[0]*sin(rbe[2])

    return np.array([x,y,z])