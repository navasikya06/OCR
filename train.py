import data_helper
from config import config
import CTC as model_keras
import numpy as np
from time import time
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, TensorBoard

def main():
    print("Loading training data...")
    model = model_keras.cnn_ctc(training=True, input_shape=(config.INPUT_HEIGHT, config.INPUT_WIDTH, config.CHANNEL), size_voc=config.SIZE_VOC)

    train_generator = data_helper.MY_Generator(config.TRAIN_LOC, config.BATCH_SIZE,
                                               (config.INPUT_HEIGHT, config.INPUT_WIDTH, config.CHANNEL))
    val_generator = data_helper.MY_Generator(config.VAL_LOC, config.BATCH_SIZE,
                                             (config.INPUT_HEIGHT, config.INPUT_WIDTH, config.CHANNEL))
    print("Finished loading training data.")

    # model.load_weights(config.CHECKPOINT_PATH, by_name=True, skip_mismatch=True)
    # model_checkpoint = ModelCheckpoint(config.CHECKPOINT_PATH,monitor='loss',verbose=1,save_best_only=True)

    # Save the model according to the conditions
    checkpoint = ModelCheckpoint(config.CHECKPOINT_PATH, monitor='loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    lr_schedule = lambda epoch: config.LEARNING_RATE * 0.1 ** (epoch // 10)
    learning_rate = np.array([lr_schedule(i) for i in range(config.NUM_EPOCH)])
    changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    tensor_board = TensorBoard(log_dir=f"./logs/{time()}")
    # Train the model
    history = model.fit(
        x=train_generator,
        validation_data=val_generator,
        epochs=config.NUM_EPOCH,
        callbacks=[checkpoint, early, changelr],
        workers=6,
        max_queue_size=12
    )

if __name__ == "__main__":
    main()
