from models.model_plain import ModelPlain

class ModelAMF(ModelPlain):
    """Train with two inputs (L, C) and with pixel loss"""

    def feed_data(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        self.AMF = data['AMF'].to(self.device)
        if need_H:
            self.H = data['H'].to(self.device)

    # ----------------------------------------
    # feed (L, C) to netG and get E
    # ----------------------------------------
    def netG_forward(self):
        self.E = self.netG(self.L, self.AMF, self.H)