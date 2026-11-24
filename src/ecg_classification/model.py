"""Eight-class CNN architecture, training and batched predictions."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .data import ImageSample, batches, image_arrays


def create_model(
    *, input_size: tuple[int, int] = (64, 64), regularization: float = 0.0001
):
    """Build the six-convolution 64/128/256 CNN with regularization and dropout."""
    from tensorflow.keras import Sequential, regularizers
    from tensorflow.keras.layers import (
        BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
    )

    model = Sequential([
        Input(shape=(input_size[1], input_size[0], 1)),
        Conv2D(64, (4, 4), padding="same", activation="relu",
               kernel_regularizer=regularizers.l1_l2(regularization, regularization)),
        BatchNormalization(),
        Conv2D(64, (4, 4), padding="same", activation="relu",
               kernel_regularizer=regularizers.l2(regularization)),
        MaxPooling2D(), Dropout(0.20),
        Conv2D(128, (4, 4), padding="same", activation="relu",
               kernel_regularizer=regularizers.l2(regularization)),
        Conv2D(128, (4, 4), padding="same", activation="relu",
               kernel_regularizer=regularizers.l2(regularization)),
        MaxPooling2D(), Dropout(0.20),
        Conv2D(256, (4, 4), padding="same", activation="relu",
               kernel_regularizer=regularizers.l2(regularization)),
        Conv2D(256, (4, 4), padding="same", activation="relu",
               kernel_regularizer=regularizers.l2(regularization)),
        MaxPooling2D(), Dropout(0.20),
        Flatten(),
        Dense(512, activation="relu", kernel_regularizer=regularizers.l2(regularization)),
        Dropout(0.5), Dense(8, activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def train_model(
    train: Sequence[ImageSample], validation: Sequence[ImageSample],
    *, epochs: int = 30, batch_size: int = 32,
    checkpoint: Path = Path("cnn.h5"), augmentation: bool = True,
    seed: int = 0,
):
    """Fit the CNN using a record-disjoint validation partition."""
    from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
    from tensorflow.keras.utils import set_random_seed

    if not train or not validation or epochs <= 0:
        raise ValueError("Training and validation must be nonempty; epochs must be positive")
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    # Set Python/NumPy/TensorFlow random generators; some GPU kernels can
    # remain nondeterministic even when a seed is specified.
    set_random_seed(seed)
    model = create_model()
    model.fit(
        batches(train, batch_size, augment=augmentation, seed=seed),
        steps_per_epoch=math.ceil(len(train) / batch_size), epochs=epochs,
        validation_data=batches(validation, batch_size, augment=False),
        validation_steps=math.ceil(len(validation) / batch_size),
        callbacks=[
            ModelCheckpoint(str(checkpoint), monitor="val_loss", save_best_only=True),
            TensorBoard(log_dir=str(checkpoint.parent / "logs"), histogram_freq=1),
        ],
    )
    return model


def predict_model(model, test: Sequence[ImageSample], *, batch_size: int = 32) -> np.ndarray:
    """Evaluate all test images deterministically, using bounded memory."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    results: list[np.ndarray] = []
    for offset in range(0, len(test), batch_size):
        image_batch, _ = image_arrays(test[offset:offset + batch_size], augment=False)
        results.append(np.asarray(model.predict(image_batch, verbose=0)))
    return np.concatenate(results) if results else np.empty((0, 8), dtype="float32")
