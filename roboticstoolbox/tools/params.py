_params = {
    "unicode": True,
}


def rtb_set_param(param, value):
    _params[param] = value

def rtb_get_param(param):
    return _params[param]