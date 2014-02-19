# Fill in the TODOs in this exercise, then run
# python 01_grad.py to see if your solution works!
#
from theano import tensor as T

def grad_sum(x, y, z):
    """
    x: A theano variable
    y: A theano variable
    z: A theano expression involving x and y

    Returns dz / dx + dz / dy
    """

    return sum(T.grad(z, [x, y]))



if __name__ == "__main__":
    x = T.scalar()
    y = T.scalar()
    z = x + y
    s = grad_sum(x, y, z)
    assert s.eval({x: 0, y:0}) == 2
    print "SUCCESS!"
