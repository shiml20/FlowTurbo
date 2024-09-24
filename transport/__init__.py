from .transport import Transport, ModelType, WeightType, PathType, Sampler

def create_transport(
    path_type='Linear'
):
    """function for creating Transport object
    **Note**: model prediction defaults to velocity
    Args:
    - path_type: type of path to use; default to linear
    """
    
    model_type = ModelType.VELOCITY

    loss_type = WeightType.NONE

    path_choice = { "Linear": PathType.LINEAR }

    path_type = path_choice[path_type]

    # create flow state
    state = Transport(
        model_type=model_type,
        path_type=path_type,
        loss_type=loss_type,
    )
    
    return state