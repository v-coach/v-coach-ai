from __future__ import annotations

import tensorflow as tf


def build_food101_model(
    image_size: int,
    num_classes: int,
    dropout_rate: float,
    imagenet_weights: bool,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    backbone = tf.keras.applications.MobileNetV3Small(
        include_top=False,
        weights="imagenet" if imagenet_weights else None,
        include_preprocessing=True,
        input_shape=(image_size, image_size, 3),
    )
    backbone.trainable = False

    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="food_probabilities")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="food101_classifier")


def compile_model(model: tf.keras.Model, learning_rate: float, top_k: int) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="top1_accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=top_k, name=f"top{top_k}_accuracy"),
        ],
    )


def unfreeze_backbone(model: tf.keras.Model, fine_tune_at_layer: int) -> None:
    backbone = _get_backbone(model)
    backbone.trainable = True
    for layer in backbone.layers[:fine_tune_at_layer]:
        layer.trainable = False


def _get_backbone(model: tf.keras.Model) -> tf.keras.Model:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "MobileNetV3" in layer.name:
            return layer
    raise ValueError("MobileNetV3 backbone not found")

