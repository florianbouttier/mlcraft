import logging

from mlcraft.utils.logging import configure_logging, get_logger, set_verbosity


def test_logging_configuration_and_retrieval():
    logger = configure_logging(verbose=1, logger_name="mlcraft.test")
    assert logger.level <= logging.INFO
    retrieved = get_logger("test")
    assert retrieved.name == "mlcraft.test"


def test_set_verbosity_updates_logger_level():
    configure_logging(verbose=0, logger_name="mlcraft.dynamic")
    logger = set_verbosity(verbose=3, logger_name="mlcraft.dynamic")
    assert logger.level == logging.DEBUG

