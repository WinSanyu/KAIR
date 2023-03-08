from models.model_plain import ModelPlain

class ModelPnP(ModelPlain):
    """Only test.
    input is self.L.
    self.H used to get checkpoint
    """

    # ----------------------------------------
    # feed (L, H) to netG and get E
    # ----------------------------------------
    def netG_forward(self):
        self.E = self.netG(self.L, self.H)