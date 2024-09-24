def none_or_str(value):
    if value == 'None':
        return None
    return value

def parse_transport_args(parser):
    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="Linear", choices=["Linear"])
    group.add_argument("--sample-eps", type=float)
    group.add_argument("--train-eps", type=float)