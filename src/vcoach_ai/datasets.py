from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from vcoach_ai.food101_labels import FOOD101_LABELS


def load_food101_datasets(config: dict) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    source = config["data"]["source"]
    image_size = config["data"]["image_size"]
    batch_size = config["data"]["batch_size"]

    if source == "tfds":
        return _load_tfds_food101(config, image_size, batch_size)
    if source == "food101_archive":
        return _load_archive_food101(config, image_size, batch_size)
    if source == "directory":
        return _load_directory_food101(config, image_size, batch_size)
    raise ValueError(f"Unsupported data source: {source}")


def _load_tfds_food101(
    config: dict,
    image_size: int,
    batch_size: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    import tensorflow_datasets as tfds

    train_ds, val_ds = tfds.load(
        config["data"]["tfds_name"],
        split=["train", "validation"],
        as_supervised=True,
        data_dir=config["data"]["tfds_data_dir"],
        shuffle_files=True,
    )

    train_ds = _prepare_dataset(train_ds, image_size, batch_size, shuffle=True)
    val_ds = _prepare_dataset(val_ds, image_size, batch_size, shuffle=False)
    return train_ds, val_ds


def _load_directory_food101(
    config: dict,
    image_size: int,
    batch_size: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        Path(config["data"]["local_train_dir"]),
        labels="inferred",
        label_mode="int",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        Path(config["data"]["local_val_dir"]),
        labels="inferred",
        label_mode="int",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
    )
    return _prepare_batched_dataset(train_ds), _prepare_batched_dataset(val_ds)


def _load_archive_food101(
    config: dict,
    image_size: int,
    batch_size: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    archive_root = Path(config["data"]["archive_root"])
    train_paths, train_labels = _read_food101_split(archive_root, split="train")
    val_paths, val_labels = _read_food101_split(archive_root, split="test")

    train_ds = _dataset_from_paths(train_paths, train_labels, image_size, batch_size, shuffle=True)
    val_ds = _dataset_from_paths(val_paths, val_labels, image_size, batch_size, shuffle=False)
    return train_ds, val_ds


def _read_food101_split(archive_root: Path, split: str) -> tuple[list[str], np.ndarray]:
    split_file = archive_root / "meta" / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Food-101 split file not found: {split_file}")

    label_to_index = {label: index for index, label in enumerate(FOOD101_LABELS)}
    image_paths = []
    labels = []

    for line in split_file.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item:
            continue

        class_name = item.split("/")[0]
        if class_name not in label_to_index:
            raise ValueError(f"Unknown Food-101 class in split file: {class_name}")

        image_paths.append(str((archive_root / "images" / f"{item}.jpg").resolve()))
        labels.append(label_to_index[class_name])

    return image_paths, np.asarray(labels, dtype="int32")


def _dataset_from_paths(
    image_paths: list[str],
    labels: np.ndarray,
    image_size: int,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if shuffle:
        dataset = dataset.shuffle(4096, reshuffle_each_iteration=True)

    return (
        dataset
        .map(lambda path, label: _load_image(path, label, image_size), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def _prepare_dataset(
    dataset: tf.data.Dataset,
    image_size: int,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    if shuffle:
        dataset = dataset.shuffle(4096, reshuffle_each_iteration=True)
    return (
        dataset
        .map(lambda image, label: _resize_image(image, label, image_size), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def _prepare_batched_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    return dataset.prefetch(tf.data.AUTOTUNE)


def _resize_image(image: tf.Tensor, label: tf.Tensor, image_size: int) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.int32)
    return image, label


def _load_image(path: tf.Tensor, label: tf.Tensor, image_size: int) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.int32)
    return image, label
