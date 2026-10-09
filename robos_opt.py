import os
import gc
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, STATUS_OK, SparkTrials


# =========================================================
# 1 CUSTOM ROBOS LOSS
# =========================================================
def robos_nn_loss(y_true, y_pred, a, epsilon, lam):
    """
    RoboS-NN custom bounded smooth loss function.
    """
    u = tf.abs(y_true - y_pred)
    val = a * tf.sqrt(tf.square(u) + epsilon) - a * tf.sqrt(epsilon)
    loss_val = lam * (1 - (val + 1) * tf.exp(-val))
    return tf.reduce_mean(loss_val)


# =========================================================
# 2 DATA GENERATOR
# =========================================================
def sample_generator_numeric(x_path, y_path, validation_split, batch_size):

    x = np.load(x_path)
    y = np.load(y_path)

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=validation_split, random_state=42
    )

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        .shuffle(len(x_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)) \
        .shuffle(len(x_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_data, val_data


# =========================================================
# 3 FIXED MLP ARCHITECTURE  (NO REGULARIZATION)
# =========================================================
def build_fixed_mlp():
    seq_size = 132
    units = 64
    layers = 2
    dropout = 0.2

    model = Sequential()
    model.add(Flatten(input_shape=(seq_size, 1)))

    for _ in range(layers):
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(1, activation='linear'))

    return model


# =========================================================
# 4 TRAINING FUNCTION (OPTIMIZES ONLY ROBOS LOSS PARAMS)
# =========================================================
def robos_loss_param_objective(cfg):

    # ======= OPTIMIZED LOSS PARAMETERS =======
    a = cfg['a']
    epsilon = cfg['epsilon']
    lamda = cfg['lamda']

    batch_size = 32
    validation_split = 0.2

    x_path = cfg['x_path']
    y_path = cfg['y_path']

    # Dynamic RoboS loss
    def dynamic_robos_loss(y_true, y_pred):
        return robos_nn_loss(y_true, y_pred, a=a, epsilon=epsilon, lam=lamda)

    # Prepare datasets
    train_set, val_set = sample_generator_numeric(
        x_path, y_path, validation_split, batch_size
    )

    # Build fixed MLP
    model = build_fixed_mlp()

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        amsgrad=True
    )

    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=5,
        restore_best_weights=True,
        min_delta=1e-4
    )

    model.compile(
        optimizer=opt,
        loss=dynamic_robos_loss
    )

    model.fit(
        train_set,
        epochs=50,
        validation_data=val_set,
        verbose=1,
        callbacks=[es]
    )

    # External validation
    x_vali = np.load(cfg['x_vali'])
    y_vali = np.load(cfg['y_vali'])

    val = tf.data.Dataset.from_tensor_slices(
        (x_vali, y_vali)
    ).batch(batch_size)

    val_loss = model.evaluate(val, verbose=0)

    print(
        f"Completed: RoboS(a={a:.3f}, eps={epsilon:.4f}, lamda={lamda:.3f}) "
        f"val_loss={val_loss:.6f}"
    )

    model_path = (
        f"/gpfs-home/p220127ma/Benchmark_Models/"
        f"mlp_robos_a{a:.3f}_eps{epsilon:.4f}_lam{lamda:.3f}.h5"
    )
    model.save(model_path)

    tf.keras.backend.clear_session()
    gc.collect()

    return {'loss': val_loss, 'status': STATUS_OK}


# =========================================================
# 5 SEARCH SPACE (ONLY ROBOS PARAMETERS)
# =========================================================
space = {
    'a': hp.uniform('a', 1.0, 10.0),
    'lamda': hp.uniform('lamda', 0.1, 1.0),
    'epsilon': hp.uniform('epsilon', 1e-4, 0.05),

    'x_path': '/gpfs-home/p220127ma/Robos_data/sunspot/x_train_new_sunspot.npy',
    'y_path': '/gpfs-home/p220127ma/Robos_data/sunspot/y_train_new_sunspot.npy',
    'x_vali': '/gpfs-home/p220127ma/Robos_data/sunspot/x_vali_sunspot.npy',
    'y_vali': '/gpfs-home/p220127ma/Robos_data/sunspot/y_vali_sunspot.npy'

}


# =========================================================
# 6 RUN OPTIMIZATION
# =========================================================
if __name__ == "__main__":

    trials = SparkTrials(parallelism=4)

    best = fmin(
        fn=robos_loss_param_objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials
    )

    print("\nOptimal RoboS Loss Parameters:")
    print(best)

    with open('Robos_Loss_Only_Opt_Result.pkl', 'wb') as f:
        pickle.dump(best, f)
