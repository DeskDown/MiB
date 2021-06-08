import torch
import torch.nn as nn
import torch.nn.functional as functional
from bisenet.build_BiSeNet import BiSeNet
from functools import partial, reduce
import sys
import os
sys.path.append(os.getcwd())
# from torch import distributed
# import inplace_abn
# from inplace_abn import InPlaceABNSync, InPlaceABN, ABN


# import models
# from modules import DeeplabV3


def make_model(opts=None, classes=None):
    # if opts.norm_act == 'iabn_sync':
    #     norm = partial(InPlaceABNSync, activation="leaky_relu", activation_param=.01)
    # elif opts.norm_act == 'iabn':
    #     norm = partial(InPlaceABN, activation="leaky_relu", activation_param=.01)
    # elif opts.norm_act == 'abn':
    #     norm = partial(ABN, activation="leaky_relu", activation_param=.01)
    # else:
    #     norm = nn.BatchNorm2d  # not synchronized, can be enabled with apex

    # body = models.__dict__[f'net_{opts.backbone}'](norm_act=norm, output_stride=opts.output_stride)
    # if not opts.no_pretrained:
    #     pretrained_path = f'pretrained/{opts.backbone}_{opts.norm_act}.pth.tar'
    #     pre_dict = torch.load(pretrained_path, map_location='cpu')
    #     del pre_dict['state_dict']['classifier.fc.weight']
    #     del pre_dict['state_dict']['classifier.fc.bias']

    #     body.load_state_dict(pre_dict['state_dict'])
    #     del pre_dict  # free memory

    # head_channels = 128

    # head = DeeplabV3(body.out_channels, head_channels, 256, norm_act=norm,
    #                  out_stride=opts.output_stride, pooling_size=opts.pooling)
    return IncrementalBiseNet(classes=classes)
    # if classes is not None:
    #     return IncrementalBiseNet(classes=classes)
    # else:
    #     pass
    #     # model = SegmentationModule(opts.num_classes, opts.fusion_mode)

    # return model


class IncrementalBiseNet(nn.Module):

    def __init__(self, classes, ncm=False, fusion_mode="mean"):
        super(IncrementalBiseNet, self).__init__()
        self.core = BiSeNet('resnet50')
        channels_out = 128
        channels_sv1 = 1024
        channels_sv2 = 2048
        # classes must be a list where [n_class_task[i] for i in tasks]
        assert isinstance(classes, list), \
            "Classes must be a list where to every index correspond the num of classes for that task"
        # classifiers for final output
        self.cls = nn.ModuleList(
            [nn.Conv2d(channels_out, c, 1) for c in classes]
        )
        # classifiers for supervision1
        self.sv1 = nn.ModuleList(
            [nn.Conv2d(channels_sv1, c, 1) for c in classes]
        )
        # classifiers for supervision2
        self.sv2 = nn.ModuleList(
            [nn.Conv2d(channels_sv2, c, 1) for c in classes]
        )

        self.classes = classes
        self.tot_classes = reduce(lambda a, b: a + b, self.classes)
        self.means = None

    def _network(self, x, ret_intermediate=False):

        # x_b = self.body(x)
        # x_pl = self.head(x_b)
        features_ffm, features_cx1, features_cx2 = self.core(x)
        self.features = features_ffm  # save features for dist. loss
        out = []
        out_cx1 = []
        out_cx2 = []

        for mod in self.cls:
            out.append(mod(features_ffm))
        x_o = torch.cat(out, dim=1)

        if self.training == False:
            return x_o

        for mod in self.sv1:
            out_cx1.append(mod(features_cx1))
        out_cx1 = torch.cat(out_cx1, dim=1)

        for mod in self.sv2:
            out_cx2.append(mod(features_cx2))
        out_cx2 = torch.cat(out_cx2, dim=1)

        # if ret_intermediate:
        #     return x_o, x_b,  x_pl
        return x_o, out_cx1, out_cx2


    def init_new_classifier(self, device):
        # Update the Final classifier
        cls = self.cls[-1]
        imprinting_w = self.cls[0].weight[0]
        bkg_bias = self.cls[0].bias[0]
        bias_diff = torch.log(torch.FloatTensor(
            [self.classes[-1] + 1])).to(device)
        new_bias = (bkg_bias - bias_diff)
        cls.weight.data.copy_(imprinting_w)
        cls.bias.data.copy_(new_bias)
        self.cls[0].bias[0].data.copy_(new_bias.squeeze(0))
        # Update the aux. output 1
        sv1 = self.sv1[-1]
        imprinting_w = self.sv1[0].weight[0]
        bkg_bias = self.sv1[0].bias[0]
        bias_diff = torch.log(torch.FloatTensor(
            [self.classes[-1] + 1])).to(device)
        new_bias = (bkg_bias - bias_diff)
        sv1.weight.data.copy_(imprinting_w)
        sv1.bias.data.copy_(new_bias)
        self.sv1[0].bias[0].data.copy_(new_bias.squeeze(0))
        # Update the aux. output 2
        sv2 = self.sv2[-1]
        imprinting_w = self.sv2[0].weight[0]
        bkg_bias = self.sv2[0].bias[0]
        bias_diff = torch.log(torch.FloatTensor(
            [self.classes[-1] + 1])).to(device)
        new_bias = (bkg_bias - bias_diff)
        sv2.weight.data.copy_(imprinting_w)
        sv2.bias.data.copy_(new_bias)
        self.sv2[0].bias[0].data.copy_(new_bias.squeeze(0))


    def forward(self, x, scales=None, do_flip=False, ret_intermediate=False):
        out_size = x.shape[-2:]
        if self.training:
            out, out_sv1, out_sv2 = self._network(x)
            out = functional.interpolate(out, scale_factor=8, mode="bilinear")
            out_sv1 = functional.interpolate(
                out_sv1, size=out_size, mode="bilinear", align_corners=False)
            out_sv2 = functional.interpolate(
                out_sv2, size=out_size, mode="bilinear", align_corners=False)
            return out, out_sv1, out_sv2

        # When testing
        return functional.interpolate(self._network(x), scale_factor=8, mode="bilinear")
        # out = self._network(x)

        # sem_logits = out[0] if ret_intermediate else out

        # sem_logits = functional.interpolate(sem_logits, size=out_size, mode="bilinear", align_corners=False)

        # if ret_intermediate:
        #     return sem_logits, {"body": out[1], "pre_logits": out[2]}

        # return sem_logits, {}

    def fix_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, inplace_abn.ABN):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


if __name__ == "__main__":
    model = make_model(classes=[2, 3])
    print(model)
    model.init_new_classifier("cpu")
