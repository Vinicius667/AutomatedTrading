import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping



class PrintCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("Training is started!")
        # Show trainer parameters
        for key, value in trainer.__dict__.items():
            if type(value) in [int, float, str] and not key.startswith("_"):
                print(f"{key}: {value}")
        print("=*" * 30)

    def on_train_end(self, trainer, pl_module):
        print("\nTraining is ended!")


print_callback = PrintCallback()
