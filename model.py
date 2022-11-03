import torch
import torch.nn as nn
from torchsummary import summary


def conv0(in_features, out_features=64, act_fn=nn.ReLU):
    return nn.ModuleList([
        nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False),
        nn.BatchNorm2d(out_features),
        act_fn(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    ])


def conv_x(in_features, mid_features, out_features, final_stride=1, act_fn=nn.ReLU):
    return nn.ModuleList([
        nn.Conv2d(
            in_channels=in_features,
            out_channels=mid_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False),
        nn.BatchNorm2d(mid_features),
        act_fn(),
        nn.Conv2d(in_channels=mid_features,
                  out_channels=mid_features,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  bias=False),
        nn.BatchNorm2d(mid_features),
        act_fn(),
        nn.Conv2d(in_channels=mid_features,
                  out_channels=out_features,
                  kernel_size=1,
                  stride=final_stride,
                  padding=0,
                  bias=False),
        nn.BatchNorm2d(out_features)
    ])


def skip_x(in_features, out_features, final_stride=1):
    return nn.ModuleList([
        nn.Conv2d(in_channels=in_features,
                  out_channels=out_features,
                  kernel_size=1,
                  stride=final_stride,
                  padding=0,
                  bias=False),
        nn.BatchNorm2d(out_features)
    ])


class ResNet50(nn.Module):
    def __init__(self, num_channels=3, classes=10, act_fn=nn.ReLU):
        super(ResNet50, self).__init__()

        self.num_channels = num_channels
        self.classes = classes

        block_repeats_lis = [3, 4, 6, 3]
        in_features_lis = [64, 256, 512, 1024]
        mid_features_lis = [64, 128, 256, 512]
        scale = 4

        self.blocks = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        self.actfn_layers = nn.ModuleList()
        self.classifier_block = nn.ModuleList()

        self.conv0_block = conv0(num_channels, act_fn=act_fn)

        for repeat, in_features, mid_features in zip(block_repeats_lis, in_features_lis, mid_features_lis):
            self.blocks.append(conv_x(in_features, mid_features, scale * mid_features, act_fn=act_fn))
            self.skip_layers.append(skip_x(in_features, scale * mid_features))
            self.actfn_layers.append(act_fn())
            for i in range(repeat - 2):
                self.blocks.append(conv_x(scale * mid_features, mid_features, scale * mid_features, final_stride=1, act_fn=act_fn))
                self.skip_layers.append(nn.ModuleList([nn.Identity()]))
                self.actfn_layers.append(act_fn())
            self.blocks.append(conv_x(scale * mid_features, mid_features, scale * mid_features, final_stride=2, act_fn=act_fn))
            self.skip_layers.append(skip_x(mid_features * scale, mid_features * scale, final_stride=2))
            self.actfn_layers.append(act_fn())

        flat_image_size = mid_features_lis[-1] * scale

        self.classifier_block.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier_block.append(nn.Flatten(start_dim=1))
        self.classifier_block.append(nn.Linear(flat_image_size, 1000))
        self.classifier_block.append(nn.Linear(1000, self.classes))
        self.classifier_block.append(nn.Softmax(dim=1))

    def forward(self, x):
        x_ = x

        for module in self.conv0_block:
            x_ = module(x_)

        for index, (block, skip_layer, actfn) in enumerate(zip(self.blocks, self.skip_layers, self.actfn_layers)):
            interm_x_ = x_.clone()

            for module in block:
                interm_x_ = module(interm_x_)

            skip_x_ = x_.clone()

            for skip_module in skip_layer:
                skip_x_ = skip_module(skip_x_)

            x_ = interm_x_ + skip_x_

            x_ = actfn(x_)

        for layer in self.classifier_block:
            x_ = layer(x_)

        return x_


if __name__ == "__main__":
    model = ResNet50(num_channels=3).to("cuda")
    model(torch.rand((1, 3, 224, 224)).to("cuda"))
    summary(model, (3, 224, 224))
