from models.model_plain import ModelPlain

class ModelAMF(ModelPlain):
    """Train with two inputs (L, C) and with pixel loss"""

    def __init__(self, opt):
        super(ModelAMF, self).__init__(opt)
        self.checkpoint = True

    def feed_data(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        self.AMF = data['AMF'].to(self.device)
        if need_H:
            self.H = data['H'].to(self.device)

    def disable_checkpoint(self):
        self.checkpoint = False

    # ----------------------------------------
    # feed (L, C) to netG and get E
    # ----------------------------------------
    def netG_forward(self):
        if self.checkpoint:
            self.E = self.netG(self.L, self.AMF, self.H)
        else:
            self.E = self.netG(self.L, self.AMF)