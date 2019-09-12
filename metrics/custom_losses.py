import torch
from torch import nn
import torch.nn.functional as F
from metrics.ssim import ssim
from metrics.tv_loss import TVLoss
#import models.networks as networks
from metrics.my_ssim import ssim_loss


# class CSSIM(nn.Module):  # Complementary SSIM
#     def __init__(self, default_range=1, filter_size=11, k1=0.01, k2=0.03, sigma=1.5, reduction='mean'):
#         super().__init__()
#         self.max_val = default_range
#         self.filter_size = filter_size
#         self.k1 = k1
#         self.k2 = k2
#         self.sigma = sigma
#         self.reduction = reduction

#     def forward(self, input, target, max_val=None):
#         max_val = self.max_val if max_val is None else max_val
#         return 1 - ssim(input, target, max_val=max_val, filter_size=self.filter_size,
#                              sigma=self.sigma, reduction=self.reduction)


# class CSSIM(nn.Module):  # Replace this with a system of summing losses in Model Trainer later on.
#     def __init__(self, default_range=1, filter_size=11, k1=0.01, k2=0.03, sigma=1.5, reduction='mean'):
#         super().__init__()
#         self.max_val = default_range
#         self.filter_size = filter_size
#         self.k1 = k1
#         self.k2 = k2
#         self.sigma = sigma
#         self.reduction = reduction

#     def forward(self, input, target, max_val=None):
#         max_val = self.max_val if max_val is None else max_val
#         input = input.unsqueeze(1)
#         target = target.unsqueeze(1)
#         ssim_value = ssim(input, target, max_val=max_val, filter_size=self.filter_size, sigma=self.sigma, reduction=self.reduction)

       
#         return ssim_value #+ self.l1_weight * l1_loss

class CSSIM(nn.Module):  # Complementary SSIM
    def __init__(self, default_range=1, filter_size=11, k1=0.01, k2=0.03, sigma=1.5, reduction='mean'):
        super().__init__()
        self.max_val = default_range
        self.filter_size = filter_size
        self.k1 = k1
        self.k2 = k2
        self.sigma = sigma
        self.reduction = reduction

    def forward(self, input, target, max_val=None):
        max_val = self.max_val if max_val is None else max_val
        input = input.unsqueeze(1)
        print (input.max())
        target = target.unsqueeze(1)
        return 1- ssim_loss(input, target, max_val=max_val, filter_size=self.filter_size, k1=self.k1, k2=self.k2,
                             sigma=self.sigma, reduction=self.reduction)

class L1CSSIM(nn.Module):  # Replace this with a system of summing losses in Model Trainer later on.
    def __init__(self, l1_weight, default_range=1, filter_size=11, k1=0.01, k2=0.03, sigma=1.5, reduction='mean'):
        super().__init__()
        self.l1_weight = l1_weight
        self.max_val = default_range
        self.filter_size = filter_size
        self.k1 = k1
        self.k2 = k2
        self.sigma = sigma
        self.reduction = reduction

    def forward(self, input, target, max_val=None):
        max_val = self.max_val if max_val is None else max_val

        cssim = 1 - ssim(input, target, max_val=max_val, filter_size=self.filter_size, sigma=self.sigma, reduction=self.reduction)

        l1_loss = F.l1_loss(input, target, reduction=self.reduction)

        return cssim + self.l1_weight * l1_loss


class L1CSSIMTV(nn.Module):  # Replace this with a system of summing losses in Model Trainer later on.
    def __init__(self, l1_weight, default_range=1, filter_size=11, k1=0.01, k2=0.03, sigma=1.5, reduction='mean', tvloss_weight=1e-4, p=2):
        super().__init__()
        self.l1_weight = l1_weight
        self.max_val = default_range
        self.filter_size = filter_size
        self.k1 = k1
        self.k2 = k2
        self.sigma = sigma
        self.reduction = reduction
        self.tvloss_weight = tvloss_weight
        self.p = p

    def forward(self, input, target, max_val=None):
        max_val = self.max_val if max_val is None else max_val

        cssim = 1 - ssim(input, target, max_val=max_val, filter_size=self.filter_size, sigma=self.sigma, reduction=self.reduction)

        l1_loss = F.l1_loss(input, target, reduction=self.reduction)
        
        tv_loss = TVLoss(input, self.tvloss_weight, self.p)

        return cssim + self.l1_weight * l1_loss + tv_loss


class C1CSSIMTV(nn.Module):  # Replace this with a system of summing losses in Model Trainer later on.
    def __init__(self, l1_weight, default_range=1, filter_size=11, k1=0.01, k2=0.03, sigma=1.5, reduction='mean', tvloss_weight=1e-4, p=2):
        super().__init__()
        self.l1_weight = l1_weight
        self.max_val = default_range
        self.filter_size = filter_size
        self.k1 = k1
        self.k2 = k2
        self.sigma = sigma
        self.reduction = reduction
        self.tvloss_weight = tvloss_weight
        self.p = p
        self.cham = CharbonnierLoss()

    def forward(self, input, target, max_val=None):
        max_val = self.max_val if max_val is None else max_val

        cssim = 1 - ssim(input, target, max_val=max_val, filter_size=self.filter_size, sigma=self.sigma, reduction=self.reduction)

        l1_loss = self.cham(input, target)
        
        tv_loss = TVLoss(input, self.tvloss_weight, self.p)

        return cssim + self.l1_weight * l1_loss + tv_loss

class ECSSIMTV(nn.Module):  # Replace this with a system of summing losses in Model Trainer later on.
    def __init__(self, l1_weight, default_range=1, filter_size=11, k1=0.01, k2=0.03, sigma=1.5, reduction='mean', tvloss_weight=1e-4, p=2):
        super().__init__()
        self.l1_weight = l1_weight
        self.max_val = default_range
        self.filter_size = filter_size
        self.k1 = k1
        self.k2 = k2
        self.sigma = sigma
        self.reduction = reduction
        self.tvloss_weight = tvloss_weight
        self.p = p
        self.ElasticLoss = ElasticLoss()

    def forward(self, input, target, max_val=None):
        max_val = self.max_val if max_val is None else max_val

        cssim = 1 - ssim(input, target, max_val=max_val, filter_size=self.filter_size, sigma=self.sigma, reduction=self.reduction)

        l1_loss = self.ElasticLoss(input, target)
        
        tv_loss = TVLoss(input, self.tvloss_weight, self.p)

        return cssim + self.l1_weight * l1_loss + tv_loss, cssim, tv_loss

## Combination loss for SRRaGAN 
class SRRaGAN(nn.Module):  
    def __init__(self, elastic_weight = 1):
        super().__init__()        
        self.cri_pix = ElasticLoss().to(self.device)        # Pixel Loss
        self.cri_fea = ElasticLoss().to(self.device)        # Feature Loss 
        self.netF = networks.define_F(opt, use_bn=False).to(self.device)

    def forward(self, input, target, max_val=None):
        max_val = self.max_val if max_val is None else max_val
        cssim = 1 - ssim(input, target, max_val=max_val, filter_size=self.filter_size, sigma=self.sigma, reduction=self.reduction)

        return 



class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        b, c, h, w = y.size()
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss/(c*b*h*w)

def LoG(imgHF):
    
    weight = [
        [0, 0, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [1, 2, -16, 2, 1],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0]
    ]
    weight = np.array(weight)

    weight_np = np.zeros((1, 1, 5, 5))
    weight_np[0, 0, :, :] = weight
    weight_np = np.repeat(weight_np, imgHF.shape[1], axis=1)
    weight_np = np.repeat(weight_np, imgHF.shape[0], axis=0)

    weight = torch.from_numpy(weight_np).type(torch.FloatTensor).to('cuda:0')
    
    return nn.functional.conv2d(imgHF, weight, padding=1)

class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size=15, sigma=3, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        return self.conv(input, weight=self.weight, groups=self.groups)

# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp, \
            grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss

    
class HFENL1Loss(nn.Module):
    def __init__(self): 
        super(HFENL1Loss, self).__init__()

    def forward(self, input, target):
        c = input.shape[1]
        smoothing = GaussianSmoothing(c, 5, 1)
        smoothing = smoothing.to('cuda:0')
        input_smooth = nn.functional.pad(input, (2, 2, 2, 2), mode='reflect')
        target_smooth = nn.functional.pad(target, (2, 2, 2, 2), mode='reflect')
        
        input_smooth = smoothing(input_smooth)
        target_smooth = smoothing(target_smooth)

        return torch.abs(LoG(input_smooth-target_smooth)).sum()
    
    
class HFENL2Loss(nn.Module):
    def __init__(self): 
        super(HFENL2Loss, self).__init__()

    def forward(self, input, target):
        c = input.shape[1]
        smoothing = GaussianSmoothing(c, 5, 1)
        smoothing = smoothing.to('cuda:0') 
        input_smooth = nn.functional.pad(input, (2, 2, 2, 2), mode='reflect')
        target_smooth = nn.functional.pad(target, (2, 2, 2, 2), mode='reflect')
        
        input_smooth = smoothing(input_smooth)
        target_smooth = smoothing(target_smooth)

        return torch.sum(torch.pow((LoG(input_smooth-target_smooth)), 2))


class ElasticLoss(nn.Module):
    def __init__(self, a=0.2): #a=0.5 default
        super(ElasticLoss, self).__init__()
        self.alpha = torch.FloatTensor([a, 1 - a]).to('cuda:0')

    def forward(self, input, target):
        if not isinstance(input, tuple):
            input = (input,)

        for i in range(len(input)):
            l2 = nn.functional.mse_loss(input[i].squeeze(), target.squeeze()).mul(self.alpha[0])
            l1 = nn.functional.l1_loss(input[i].squeeze(), target.squeeze()).mul(self.alpha[1])
            loss = l1 + l2

        return loss