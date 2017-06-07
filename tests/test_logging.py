import pytest
import skippylab.loggers as log


@pytest.fixture(params=[(10, 0),\
                        (20, 0),\
                        (30, 0)])

def prepare_test_data_for_logger(request):
    return request.param

# test logger
def test_logger(prepare_test_data_for_logger):
    import logging

    loglevel, __ = prepare_test_data_for_logger
    assert isinstance(log.get_logger(loglevel), logging.Logger)


