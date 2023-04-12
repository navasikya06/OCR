class Config():
    def __init__(self):
        self.USE_GPU = True

        self.CHARACTER_HEIGHT = 32
        self.CHARACTER_WIDTH = 32
        self.CHARACTER_STEP = 2
        self.WINDOWS = [32]

        self.INPUT_HEIGHT = 32
        self.INPUT_WIDTH = 256
        self.CHANNEL = 3
        self.MAX_SIZE_TEXT = 17
        self.SIZE_VOC = 2842

        self.BATCH_SIZE = 16
        self.NUM_EPOCH = 500
        self.LEARNING_RATE = 0.001

        self.TRAIN_LOC = '/home/tchu/Desktop/result/'
        self.VAL_LOC = '/home/tchu/Desktop/test'
        self.TEST_LOC = '/home/tchu/Desktop/test'
        self.CHECKPOINT_PATH = "checkpoints/vin.hdf5"
        self.VOC_PATH = "checkpoints/voc.txt"

config = Config()
