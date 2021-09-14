from pytorch_msssim import SSIM
import torch.nn as nn
import torch.nn.functional as F


class reconLoss(nn.Module):
    """
        alpha : 0 to 1 (float value) 
                1 : for MSE
                0 : for SSIM
    """

    def __init__(self, in_channels=1, use_ssim=True, alpha=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.alpha = alpha
        self.MSE = nn.MSELoss()
        self.use_ssim = use_ssim

        if use_ssim:
            self.SSIM = SSIM(
                data_range=1.0, channel=self.in_channels, size_average=True)

    def forward(self, recon, input_):
        mse = self.MSE(recon, input_)
        if self.use_ssim:
            ssim = self.SSIM(recon, input_)

            mse = self.alpha * mse + (1-self.alpha) * ssim

        return mse


class NCELoss(nn.Module):
    def __init__(self, temperature=0.0009, device='cpu'):
        super().__init__()
        self.temperature = temperature
        self.device = device

    def cross_entropy(self, preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

    def forward(self, features1, features2):
        logits = (features1 @ features2.T) / self.temperature
        Feature1_similarity = features1 @ features1.T
        Feature2_similarity = features2 @ features2.T
        targets = F.softmax(
            (Feature1_similarity + Feature2_similarity) / 2 * self.temperature, dim=-1
        )
        Feature2_loss = self.cross_entropy(
            logits, targets, reduction='none').to(self.device)
        Feature1_loss = self.cross_entropy(
            logits.T, targets.T, reduction='none').to(self.device)
        loss = (Feature1_loss + Feature2_loss) / 2.0
        return loss.mean()
