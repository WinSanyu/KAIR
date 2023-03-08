from models.model_plain import ModelPlain

class ModelPnP2(ModelPlain):
    """Only test.
    self.L: noise
    self.C: preprocess
    self.H used to get checkpoint
    """

    def feed_data(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        self.C = data['C'].to(self.device)
        if need_H:
            self.H = data['H'].to(self.device)

    # ----------------------------------------
    # feed (L, C, H) to netG and get E
    # ----------------------------------------
    def netG_forward(self):
        self.E = self.netG(self.L, self.C, self.H)