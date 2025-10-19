from agent import PPO_Agent
import torch

class EWC_Agent(PPO_Agent):
    def __init__(self, *args, ewc_lambda=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewc_lambda = ewc_lambda
        self.prev_params = {}
        self.fisher = {}

    def compute_fisher(self, data):
        for name, param in self.actor.named_parameters():
            self.fisher[name] = param.grad.pow(2).clone()

    def consolidate(self):
        for name, param in self.actor.named_parameters():
            self.prev_params[name] = param.data.clone()

    def ewc_loss(self):
        loss = 0
        for name, param in self.actor.named_parameters():
            if name in self.fisher:
                fisher = self.fisher[name]
                prev_param = self.prev_params[name]
                loss += (fisher * (param - prev_param).pow(2)).sum()
        return self.ewc_lambda * loss

    def update(self, *args, **kwargs):
        base_loss = super().update(*args, **kwargs)
        if len(self.fisher) > 0:
            ewc_loss = self.ewc_loss()
            base_loss += ewc_loss
        return base_loss
