"""
Full TensorFlow script: train MNIST with a custom training loop and log
weight + gradient histograms to TensorBoard.

Run:
    python train.py
    tensorboard --logdir logs/
"""

import datetime
import tensorflow as tf

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

BATCH_SIZE = 128
train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(10_000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
test_ds = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------
def build_model():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu", name="dense_1"),
            tf.keras.layers.Dense(64, activation="relu", name="dense_2"),
            tf.keras.layers.Dense(10, name="logits"),
        ]
    )

model = build_model()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")
test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name="test_acc")

# ---------------------------------------------------------------------------
# 3. TensorBoard writers
# ---------------------------------------------------------------------------
run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_writer = tf.summary.create_file_writer(f"logs/{run_id}/train")
test_writer = tf.summary.create_file_writer(f"logs/{run_id}/test")

# ---------------------------------------------------------------------------
# 4. Training step
# ---------------------------------------------------------------------------
HIST_EVERY = 100  # log histograms every N steps (cheap & readable)

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_acc.update_state(y, logits)
    return loss, grads


def log_histograms(grads, step):
    """Log weight + gradient histograms. Called eagerly (outside @tf.function)."""
    with train_writer.as_default():
        for var, grad in zip(model.trainable_variables, grads):
            tf.summary.histogram(f"weights/{var.name}", var, step=step)
            if grad is not None:
                tf.summary.histogram(f"grads/{var.name}", grad, step=step)
                # Also log the gradient norm as a scalar — useful for debugging
                tf.summary.scalar(
                    f"grad_norm/{var.name}", tf.norm(grad), step=step
                )

# ---------------------------------------------------------------------------
# 5. Training loop
# ---------------------------------------------------------------------------
EPOCHS = 5
global_step = 0

for epoch in range(EPOCHS):
    train_acc.reset_state()
    test_acc.reset_state()

    for x, y in train_ds:
        loss, grads = train_step(x, y)

        with train_writer.as_default():
            tf.summary.scalar("loss", loss, step=global_step)

        if global_step % HIST_EVERY == 0:
            log_histograms(grads, step=global_step)

        global_step += 1

    # Eval at end of epoch
    for x, y in test_ds:
        logits = model(x, training=False)
        test_acc.update_state(y, logits)

    with train_writer.as_default():
        tf.summary.scalar("accuracy", train_acc.result(), step=epoch)
    with test_writer.as_default():
        tf.summary.scalar("accuracy", test_acc.result(), step=epoch)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}  "
        f"train_acc={train_acc.result():.4f}  "
        f"test_acc={test_acc.result():.4f}"
    )

train_writer.close()
test_writer.close()
print(f"\nDone. View with:  tensorboard --logdir logs/{run_id}")
