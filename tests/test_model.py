import pytest


def test_cnn_architecture_if_tensorflow_available():
    pytest.importorskip("tensorflow")
    from ecg_classification.model import create_model

    model = create_model()
    assert model.input_shape == (None, 64, 64, 1)
    assert model.output_shape == (None, 8)
    assert sum(layer.__class__.__name__ == "Conv2D" for layer in model.layers) == 6
    assert model.layers[-1].activation.__name__ == "softmax"
    assert model.loss == "categorical_crossentropy"


def test_predict_model_runs_in_batches_without_random_augmentation(tmp_path):
    import cv2
    import numpy as np
    from ecg_classification.data import ImageSample
    from ecg_classification.model import predict_model

    samples = []
    for index in range(3):
        image_file = tmp_path / f"test{index}.png"
        assert cv2.imwrite(str(image_file), np.full((128, 128), 70 + index, dtype="uint8"))
        samples.append(ImageSample(image_file, "NOR", "100", 0))

    class DummyModel:
        batch_sizes = []

        def predict(self, x, verbose=0):
            self.batch_sizes.append(len(x))
            assert x.shape[1:] == (64, 64, 1)
            return np.tile(np.array([[1.0] + [0.0] * 7]), (len(x), 1))

    fake_model = DummyModel()
    predictions = predict_model(fake_model, samples, batch_size=2)
    assert predictions.shape == (3, 8)
    assert fake_model.batch_sizes == [2, 1]
