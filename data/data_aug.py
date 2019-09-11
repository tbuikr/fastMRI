import torch
from torch import autograd

from data.transforms import to_tensor, normalize, normalize_instance


class InputTrainTransform:
    def __init__(self, is_training=False):
        self.is_training = is_training

    def __call__(self, ds_slice, gt_slice, attrs, file_name, s_idx, acc_fac):
        with torch.autograd.no_grad():
            ds_slice, mean, std = normalize_instance(to_tensor(ds_slice))
            gt_slice = normalize(to_tensor(gt_slice), mean, std)

            ds_slice = ds_slice.clamp(min=-6, max=6).unsqueeze(dim=0)
            gt_slice = gt_slice.clamp(min=-6, max=6).unsqueeze(dim=0)

            ds_slice, gt_slice = self.augment_data(ds_slice, gt_slice)

        return ds_slice, gt_slice, 0

    # noinspection PyTypeChecker
    def augment_data(self, ds_slice, gt_slice):
        if self.is_training:
            prob = torch.tensor(0.5)
            flip_lr = torch.rand(()) < prob
            flip_ud = torch.rand(()) < prob

            if flip_lr and flip_ud:
                ds_slice = torch.flip(ds_slice, dims=[-2, -1])
                gt_slice = torch.flip(gt_slice, dims=[-2, -1])

            elif flip_lr:
                ds_slice = torch.flip(ds_slice, dims=[-1])
                gt_slice = torch.flip(gt_slice, dims=[-1])

            elif flip_ud:
                ds_slice = torch.flip(ds_slice, dims=[-2])
                gt_slice = torch.flip(gt_slice, dims=[-2])

        return ds_slice, gt_slice


class InputTestTransform:
    def __init__(self):
        pass

    def __call__(self, ds_slice, gt_slice, attrs, file_name, s_idx, acc_fac):
        assert gt_slice is None
        with torch.autograd.no_grad():
            ds_slice, mean, std = normalize_instance(to_tensor(ds_slice))
            ds_slice = ds_slice.clamp(min=-6, max=6).unsqueeze(dim=0)

        assert isinstance(file_name, str) and isinstance(s_idx, int), 'Incorrect types!'
        extra_params = dict(mean=mean, std=std, acc_fac=acc_fac, attrs=attrs)
        return ds_slice, file_name, s_idx, extra_params