from __future__ import annotations

import argparse

import numpy as np
import tensorflow as tf

from vcoach_ai.food101_labels import FOOD101_LABELS, display_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    image = load_image(args.image, args.image_size)
    scores = model.predict(image[None, ...], verbose=0)[0]
    top_indices = np.argsort(scores)[::-1][: args.top_k]

    for rank, index in enumerate(top_indices, start=1):
        label = FOOD101_LABELS[index]
        print(f"{rank}\t{label}\t{display_name(label)}\t{scores[index]:.4f}")


def load_image(path: str, image_size: int) -> np.ndarray:
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [image_size, image_size])
    return tf.cast(image, tf.float32).numpy()


if __name__ == "__main__":
    main()

