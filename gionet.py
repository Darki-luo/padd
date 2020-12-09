from math import ceil
import paddle

from paddle.nn.functional import interpolate

from resnet_dilated import ResNet101
from paddle.nn import Conv2D, BatchNorm, Layer, Sequential, Conv1D, ReLU

print(paddle.__version__)
#paddle.disable_static()


class GCNModule(Layer):
    def __init__(self, num_channels, num_nodes):
        super(GCNModule, self).__init__()
        self.conv1 = Conv1D(num_nodes, num_nodes, kernel_size=1)
        self.relu = ReLU()
        self.conv2 = Conv1D(num_channels, num_channels, kernel_size=1)

    def forward(self, inputs):
        # inputs.shape(B,C,N）
        #print(inputs.shape)
        #x=fluid.layers.transpose(inputs, perm=(0, 2, 1))
        x=paddle.transpose(inputs, perm=[0, 2, 1])
        #print(x.shape)
        x = self.conv1(x)
        #print(x.shape)
        #x = fluid.layers.transpose(x, perm=(0,2,1))
        x = paddle.transpose(x, perm=[0,2,1])
        #print(x.shape)
        x = x - inputs
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        x = self.relu(x)
        #print(x.shape)

        return x



class GIoNet(Layer):
    def __init__(self, num_classes=59, IMG_SIZE = None,backbone='resnet50'):
        super(GIoNet, self).__init__()

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

        in_channels = 2048
        out_channels = 256
        # gionet channels 2048-> 256 num_nodes = H' *W'
        self.layer5 = Sequential(
            Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            BatchNorm(out_channels, act='relu')
        )
        num_node = ceil(IMG_SIZE[0]/8)*ceil(IMG_SIZE[1]/8)
        self.gcnmodule = GCNModule(num_channels=out_channels, num_nodes=num_node)
        self.classifier = Conv2D(in_channels=out_channels, out_channels=num_classes, kernel_size=1)
        #self.pspmoduls = PSPModule(num_channels, [1,2,3,6])
        #num_channels *= 2


    def forward(self, inputs):
        x = self.layer0(inputs)
        #print(x.shape)
        x = self.layer1(x)
        #print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape)
        x = self.layer5(x)
        #print(x.shape)
        n,c,h,w = x.shape
        #print(x.shape)
        x = paddle.reshape(x, shape=[-1, c, h*w])
        #print(x.shape)
        x = self.gcnmodule(x)
        #print(x.shape)
        x = paddle.reshape(x, shape=[-1,c,h,w])
        #print(x.shape)
        x = self.classifier(x)
        #print(x.shape)
        x = interpolate(x, inputs.shape[2::],align_corners=False) #2::指 2和3
        #print(x.shape)

        return x



# aux: tmp_x = layer3


def main():
    #x = paddle.tensor.rand((2, 3, 520, 520),dtype='float32')
    #print(x.shape)
    x = (2, 3, 520, 520)
    model = GIoNet(num_classes=59,IMG_SIZE=x[2::])
    params_info = paddle.summary(model, x)
    print(params_info)
    #model.train()
    #pred = model(x)
    #print(pred.shape)


if __name__ == "__main__":
    main()
