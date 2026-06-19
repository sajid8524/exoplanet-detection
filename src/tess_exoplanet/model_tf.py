from __future__ import annotations


def build_global_local_cnn(
    global_bins: int = 301,
    local_bins: int = 101,
    scalar_dim: int = 11,
    n_classes: int = 4,
):
    """Optional TensorFlow Astronet-style model.

    This function imports TensorFlow lazily so the rest of the pipeline can run
    in lightweight environments.
    """
    import tensorflow as tf

    def conv_branch(name: str, length: int):
        inputs = tf.keras.Input(shape=(length, 1), name=f"{name}_input")
        x = tf.keras.layers.Conv1D(32, 5, padding="same", activation="relu")(inputs)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(96, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        return inputs, x

    global_input, global_features = conv_branch("global", global_bins)
    local_input, local_features = conv_branch("local", local_bins)
    scalar_input = tf.keras.Input(shape=(scalar_dim,), name="scalar_input")
    scalar_features = tf.keras.layers.Dense(32, activation="relu")(scalar_input)

    x = tf.keras.layers.Concatenate()([global_features, local_features, scalar_features])
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=[global_input, local_input, scalar_input], outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

