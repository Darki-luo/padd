import numpy as np
import paddle

from paddle.nn.functional import interpolate

from resnet_dilated import ResNet50, ResNet101
from paddle import to_tensor#, fluid
from paddle.nn import Conv2D, BatchNorm, Dropout, Layer, Sequential, AdaptiveMaxPool2D

print(paddle.__version__)
paddle.disable_static()

# pool with different bin_size
# interpolate back to input size
# concat
class PSPModule(Layer):
    def __init__(self, num_channels, bin_size_list):
        super(PSPModule, self).__init__()
        self.bn_size_list = bin_size_list
        num_filters = num_channels // len(bin_size_list)
        self.features = []
        for i in range(len(bin_size_list)):
            self.features.append(
                Sequential(
                    AdaptiveMaxPool2D(self.bn_size_list[i]),
                    Conv2D(in_channels=num_channels,out_channels=num_filters,kernel_size=1),
                    BatchNorm(num_filters,act='relu')
                )
            )

    def forward(self, inputs):
        out = [inputs]
        for idx, f in enumerate(self.features):
            #x = fluid.layers.adaptive_pool2d(inputs,self.bn_size_list[idx])
            x = f(inputs)
            x = interpolate(x, inputs.shape[2::], align_corners=False)
            out.append(x)

        out = paddle.concat(out,axis=1)
        return out


class PSPNet(Layer):
    def __init__(self, num_classes=59, backbone='resnet50'):
        super(PSPNet, self).__init__()

        res = ResNet101(pretrained=False)
        # stem: res.conv, res.pool2d_max
        self.layer0 = Sequential(
            res.conv,
            res.pool2d_max
        )
        self.layer1 = res.layer1
        self.layer2 = res.layer2
        self.layer3 = res.layer3
        self.layer4 = res.layer4

        num_channels = 2048
        # psp: 2048 -> 2048*2
        self.pspmoduls = PSPModule(num_channels, [1,2,3,6])
        num_channels *= 2
        # cls: 2048*2 -> 512 -> num_classes
        self.classifier = Sequential(
            Conv2D(num_channels,512,kernel_size=3,padding=1),
            BatchNorm(512,act='relu'),
            Dropout(0.1),
            Conv2D(512,num_classes,kernel_size=1)
        )
        # aux: 1024 -> 256 -> num_classes

    def forward(self, inputs):
        x = self.layer0(inputs)
        print(x.shape)
        x = self.layer1(x)
        print(x.shape)
        x = self.layer2(x)
        print(x.shape)
        x = self.layer3(x)
        print(x.shape)
        x = self.layer4(x)
        print(x.shape)
        x = self.pspmoduls(x)
        print(x.shape)
        x = self.classifier(x)
        print(x.shape)
        x = interpolate(x, inputs.shape[2::],align_corners=False) #2::指 2和3

        return x



# aux: tmp_x = layer3


def main():
    x_data = np.random.rand(2, 3, 512, 512).astype(np.float32)
    x = to_tensor(x_data)
    model = PSPNet(num_classes=59)
    model.train()
    pred = model(x)
    print(pred.shape)


if __name__ == "__main__":
    main()
