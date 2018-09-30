def _check_abs_max_grad(abs_max_grad, model):
    """Checks `model` for a new largest gradient for this epoch, in order to
    track gradient explosions.
    """
    finite_grads = [p.grad.data
                    for p in model.parameters()
                    if p.grad is not None]

    new_max_grad = max([grad.max() for grad in finite_grads])
    new_min_grad = min([grad.min() for grad in finite_grads])

    new_abs_max_grad = max(new_max_grad, abs(new_min_grad))
    if new_abs_max_grad > abs_max_grad:
        print('abs max grad: ', new_abs_max_grad)
        return new_abs_max_grad

    return abs_max_grad

def _get_max_abs_weight_value(model):
    finite_weights = [p.data
                    for p in model.parameters()
                    if hasattr(p, 'weight') and p.weight is not None]

    print("Printing all parameters!")
    # for para in model.parameters():
    #     print(para.data)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)

    # for x in finite_weights:
    #     print(x)

    max_weight = max([weight.max() for weight in finite_weights])
    min_weight = min([weight.min() for weight in finite_weights])

    return max_weight, min_weight