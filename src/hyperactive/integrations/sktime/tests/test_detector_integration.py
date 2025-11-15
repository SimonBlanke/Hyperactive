"""Smoke tests for the sktime detector integration."""

def test_detector_integration_imports():
    from hyperactive.experiment.integrations import SktimeDetectorExperiment
    from hyperactive.integrations.sktime import TSDetectorOptCv

    assert SktimeDetectorExperiment is not None
    assert TSDetectorOptCv is not None
