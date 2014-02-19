from theano import tensor as T
from theano import function

def evaluate(x, y, expr, x_value, y_value):
    """
    x: A theano variable
    y: A theano variable
    expr: A theano expression involving x and y
    x_value: A numpy value
    y_value: A numpy value

    Returns the value of expr when x_value is substituted for x
    and y_value is substituted for y
    """

    return function([x, y], expr)(x_value, y_value)


if __name__ == "__main__":
    x = T.iscalar()
    y = T.iscalar()
    z = x + y
    assert evaluate(x, y, z, 1, 2) == 3
    print "SUCCESS!"
