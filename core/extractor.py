import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="batch", stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.GELU()

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(BottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes // 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes // 4, planes // 4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes // 4, planes, kernel_size=1, padding=0)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.GELU()

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes // 4)
            self.norm2 = nn.BatchNorm2d(planes // 4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes // 4)
            self.norm2 = nn.InstanceNorm2d(planes // 4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(self, base_channel=64, norm_fn="batch"):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=base_channel // 8, num_channels=base_channel)
        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(base_channel)
        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(base_channel)
        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, base_channel, kernel_size=7, stride=2, padding=3)
        # self.relu1 = nn.ReLU(inplace=True)
        self.relu1 = nn.GELU()

        self.in_planes = base_channel
        self.down_layer1 = self._make_down_layer(base_channel, stride=1)
        self.down_layer2 = self._make_down_layer(round(base_channel * 1.5), stride=2)
        self.down_layer3 = self._make_down_layer(base_channel * 2, stride=2)
        # self.down_layer4 = self._make_down_layer(round(base_channel * 2 * 1.5), stride=2)
        # self.down_layer5 = self._make_down_layer(base_channel * 2 * 2, stride=2)
        self.down_dim = self.in_planes
        # self.up_layer1 = self._make_up_layer(round(base_channel * 1.5), scale=2.0)
        # self.up_layer2 = self._make_up_layer(base_channel * 2, scale=2.0)
        # self.up_dim = self.in_planes
        # self.top_layer = \
        #     nn.Sequential(*(nn.Conv2d(base_channel * 2 * 2, round(base_channel * 2 * 1.5), kernel_size=1, padding=0),
        #                     self._get_norm_func(base_channel * 2, norm_fn=self.norm_fn)))

        # self.up_top1 = \
        #     nn.Sequential(*(nn.Conv2d(base_channel * 2, round(base_channel * 1.5), kernel_size=1, padding=0),
        #                     self._get_norm_func(round(base_channel * 1.5), norm_fn=self.norm_fn)))
        # self.up_lateral1 = \
        #     nn.Sequential(*(nn.Conv2d(round(base_channel * 1.5), round(base_channel * 1.5), kernel_size=1, padding=0),
        #                     self._get_norm_func(round(base_channel * 1.5), norm_fn=self.norm_fn)))
        # self.up_smooth1 = \
        #     nn.Sequential(*(nn.Conv2d(round(base_channel * 1.5), round(base_channel * 1.5), kernel_size=3, padding=1),
        #                     self._get_norm_func(round(base_channel * 1.5), norm_fn=self.norm_fn),
        #                     nn.GELU()))
        # self.up_top2 = \
        #     nn.Sequential(*(nn.Conv2d(round(base_channel * 1.5), base_channel, kernel_size=1, padding=0),
        #                     self._get_norm_func(base_channel, norm_fn=self.norm_fn)))
        # self.up_lateral2 = \
        #     nn.Sequential(*(nn.Conv2d(base_channel, base_channel, kernel_size=1, padding=0),
        #                     self._get_norm_func(base_channel, norm_fn=self.norm_fn)))
        # self.up_smooth2 = \
        #     nn.Sequential(*(nn.Conv2d(base_channel, base_channel, kernel_size=3, padding=1),
        #                     self._get_norm_func(base_channel, norm_fn=self.norm_fn),
        #                     nn.GELU()))
        # self.up_dim = base_channel

        # self.up_smooth2 = nn.Conv2d(base_channel * 2, base_channel * 2, kernel_size=3, padding=1)
        # self.up_lateral2 = nn.Conv2d(base_channel * 2, base_channel * 2, kernel_size=1, padding=0)
        # self.up_layer3 = self._make_up_layer(base_channel * 2, scale=2.0)
        self.up_layer1 = self._make_up_layer(base_channel, scale=2.0)
        self.up_dim = self.in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_down_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def _make_up_layer(self, dim, scale=2.0):
        layer1 = nn.Upsample(scale_factor=scale, mode="bilinear")
        layer2 = nn.Conv2d(self.in_planes, dim, kernel_size=3, padding=1)
        if self.norm_fn == "group":
            layer3 = nn.GroupNorm(num_groups=dim // 8, num_channels=dim)
        elif self.norm_fn == "batch":
            layer3 = nn.BatchNorm2d(dim)
        elif self.norm_fn == "instance":
            layer3 = nn.InstanceNorm2d(dim)
        else:
            layer3 = nn.Sequential()
        # layer4 = nn.ReLU()
        layer4 = nn.GELU()
        layers = (layer1, layer2, layer3, layer4)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def _get_norm_func(self, channel, norm_fn="batch"):

        if norm_fn == "group":
            norm = nn.GroupNorm(num_groups=channel // 8, num_channels=channel)
        elif norm_fn == "batch":
            norm = nn.BatchNorm2d(channel)
        elif norm_fn == "instance":
            norm = nn.InstanceNorm2d(channel)
        elif norm_fn == "none":
            norm = nn.Sequential()

        return norm

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        D1 = self.down_layer1(x)
        D2 = self.down_layer2(D1)
        D3 = self.down_layer3(D2)
        # D4 = self.down_layer4(D3)
        # D5 = self.down_layer5(D4)

        # D5_x1, D5_x2 = torch.split(D5, D5.shape[0] // 2, dim=0)

        # U1 = self.up_layer1(D5_x1)
        # U2 = self.up_layer2(U1)

        # D1_x1, D1_x2 = torch.split(D1, D1.shape[0] // 2, dim=0)
        # D2_x1, D2_x2 = torch.split(D2, D2.shape[0] // 2, dim=0)
        D3_x1, D3_x2 = torch.split(D3, D3.shape[0] // 2, dim=0)
        # D4_x1, D4_x2 = torch.split(D4, D4.shape[0] // 2, dim=0)
        # D5_x1, D5_x2 = torch.split(D5, D5.shape[0] // 2, dim=0)

        # T = self.top_layer(D5_x1)

        # T1 = self.up_top1(D3_x1)
        # D2_x1 = self.up_lateral1(D2_x1)
        # U1 = self.up_smooth1(F.gelu(F.upsample(T1, scale_factor=2.0, mode="bilinear") + D2_x1))
        # T2 = self.up_top2(U1)
        # D1_x1 = self.up_lateral2(D1_x1)
        # U2 = self.up_smooth2(F.gelu(F.upsample(T2, scale_factor=2.0, mode="bilinear") + D1_x1))

        U1 = self.up_layer1(D3_x1)

        return D3_x1, D3_x2, U1


class SmallEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn="batch", dropout=0.0):
        super(SmallEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)

        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(32)

        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(32)

        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        # self.relu1 = nn.ReLU(inplace=True)
        self.relu1 = nn.GELU()

        self.in_planes = 32
        self.layer1 = self._make_layer(32, stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x
