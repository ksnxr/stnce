_DATAS = {}


def register_data(cls=None, *, name=None):

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _DATAS:
            raise ValueError(f"{local_name} already registered")
        _DATAS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_data(name):
    return _DATAS[name]


def create_data(config, workdir):
    data_name = config.data.name
    data = get_data(data_name)(config, workdir)
    return data
