def pytest_addoption(parser):
    parser.addoption("--grid_shape", action="store", default=(32,)*3)
    parser.addoption("--proc_shape", action="store", default=(1,)*3)


def tuplify(string):
    if isinstance(string, str):
        return tuple(int(i) for i in string.split(','))
    else:
        return string


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    grid_shape = metafunc.config.option.grid_shape
    if 'grid_shape' in metafunc.fixturenames and grid_shape is not None:
        metafunc.parametrize("grid_shape", [tuplify(grid_shape)])

    proc_shape = metafunc.config.option.proc_shape
    if 'proc_shape' in metafunc.fixturenames and proc_shape is not None:
        metafunc.parametrize("proc_shape", [tuplify(proc_shape)])
