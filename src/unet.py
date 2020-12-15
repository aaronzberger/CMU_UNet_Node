import torch.nn as nn
import torch.nn.functional as F
import torch


class UNetConv2(nn.Module):
    def __init__(self, in_size, out_size, use_batchnorm):
        super(UNetConv2, self).__init__()

        if use_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 1),
                nn.BatchNorm2d(out_size),
                nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, 1, 1),
                nn.BatchNorm2d(out_size),
                nn.ReLU()
            )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, use_deconv):
        super(UNetUp, self).__init__()
        self.conv = UNetConv2(in_size, out_size, False)
        if use_deconv:
            self.up = nn.ConvTranspose2d(
                in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = inputs1.size()[2] - outputs2.size()[2]

        padding = 2 * [offset // 2, offset // 2]

        outputs2 = F.pad(outputs2, padding)

        return self.conv(torch.cat([inputs1, outputs2], 1))


class UNet(nn.Module):
    def __init__(
        self, feature_scale=1, output_dim=1,
        use_deconv=True, use_batchnorm=True
    ):
        super(UNet, self).__init__()
        self.use_deconv = use_deconv
        self.in_channels = 24
        self.use_batchnorm = use_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UNetConv2(
            self.in_channels, filters[0], self.use_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = UNetConv2(filters[0], filters[1], self.use_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = UNetConv2(filters[1], filters[2], self.use_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = UNetConv2(filters[2], filters[3], self.use_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = UNetConv2(filters[3], filters[4], self.use_batchnorm)

        # upsampling
        self.up_concat4 = UNetUp(filters[4], filters[3], self.use_deconv)
        self.up_concat3 = UNetUp(filters[3], filters[2], self.use_deconv)
        self.up_concat2 = UNetUp(filters[2], filters[1], self.use_deconv)
        self.up_concat1 = UNetUp(filters[1], filters[0], self.use_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], output_dim, 1)

        def init_weights(m):
            if type(m) == nn.Conv2d or type(m) == nn.Linear:
                torch.nn.init.kaiming_normal_(m.weight.data)

        self.apply(init_weights)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        final = self.final(up1)

        return final
