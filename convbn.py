class ConvBNLayer(Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        super(ConvBNLayer,self).__init__(
            Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=groups
            ),
            BatchNorm2D(num_features=out_channels)
        )



class ConvBNLayer(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, act='relu'):
        super(ConvBNLayer,self).__init__()

        self._conv = Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups
        )
        self._bn = BatchNorm2D(num_features=out_channels)
        self._relu = ReLU()
        self.act = act

    def forward(self, inputs):
        x = self._conv(inputs)
        x = self._bn(x)
        if self.act == 'relu':
            x = self._relu(x)
        return x