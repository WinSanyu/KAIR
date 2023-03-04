from models.model_plain import ModelPlain

class ModelCheckpoint(ModelPlain):
    """Train with two inputs (L, C) and with pixel loss"""

    # ----------------------------------------
    # feed (L, C) to netG and get E
    # ----------------------------------------
    def netG_forward(self):
        self.E = self.netG(self.L, self.H)