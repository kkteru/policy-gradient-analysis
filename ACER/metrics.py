import numpy as np

def grad_norm(model):
    """ Monitor Norm of gradients """
    parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def grad_var(grad_queue, curr_grad, max_len):
    """ Empirical Moving Average of Grad Variance """
    # TODO: this should be var of grad, not var of grad norm
    grad_queue.append(curr_grad)
    if len(grad_queue) > max_len:
        grad_queue.popleft()
    grad_var = np.var(list(grad_queue))
    return grad_var

def weight_norm(model):
    """ Monitor Norm of weights """
    parameters = list(filter(lambda p: p is not None, model.parameters()))
    total_norm = 0
    for p in parameters:
        param_norm = p.data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm
