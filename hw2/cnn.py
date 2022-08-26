import torch
import torch.nn as nn
import itertools as it
from typing import Sequence

ACTIVATIONS = {"relu": nn.ReLU, "lrelu": nn.LeakyReLU}
POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
        self,
        in_size,
        out_classes: int,
        channels: Sequence[int],
        pool_every: int,
        hidden_dims: Sequence[int],
        conv_params: dict = {},
        activation_type: str = "relu",
        activation_params: dict = {},
        pooling_type: str = "max",
        pooling_params: dict = {},
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.layer_inputs = []
        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()


    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)
        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ACT)*P -> POOL]*(N/P)
        #  Apply activation function after each conv, using the activation type and
        #  parameters.
        #  Apply pooling to reduce dimensions after every P convolutions, using the
        #  pooling type and pooling parameters.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ACTs should exist at the end, without a POOL after them.
        # ====== YOUR CODE: ======
        P = self.pool_every
        l = len(self.channels)
        new_in_h = in_h
        new_in_w = in_w
        for i, out_channels in enumerate(self.channels):
          layers.append(nn.Conv2d(in_channels, out_channels, *self.conv_params.values()))
          new_in_h = ((new_in_h + 2 * self.conv_params['padding'] - 1 * (self.conv_params['kernel_size'] - 1) - 1) // (self.conv_params['stride'])) + 1
          new_in_w = ((new_in_w + 2 * self.conv_params['padding'] - 1 * (self.conv_params['kernel_size'] - 1) - 1) // (self.conv_params['stride'])) + 1
          self.layer_inputs.append((new_in_h, new_in_w))
  
          layers.append(ACTIVATIONS[self.activation_type](*self.activation_params.values()))
          in_channels = out_channels
          if (i+1) % P == 0:
            layers.append(POOLINGS[self.pooling_type](*self.pooling_params.values()))
            #Assume padding is same as kernel size
            new_in_h = int((new_in_h - self.pooling_params['kernel_size']) / self.pooling_params['kernel_size']) + 1
            new_in_w = int((new_in_w - self.pooling_params['kernel_size']) / self.pooling_params['kernel_size']) + 1
            self.layer_inputs.append((new_in_h, new_in_w))

          if l%P != 0:  #at last N mod P layers
            for j in range(l%P):
              layers.append(nn.Conv2d(in_channels, out_channels, *self.conv_params.values()))
              new_in_h = ((new_in_h + 2 * self.conv_params['padding'] - 1 * (self.conv_params['kernel_size'] - 1) - 1) // (self.conv_params['stride'])) + 1
              new_in_w = ((new_in_w + 2 * self.conv_params['padding'] - 1 * (self.conv_params['kernel_size'] - 1) - 1) // (self.conv_params['stride'])) + 1
              self.layer_inputs.append((new_in_h, new_in_w))
              
              layers.append(ACTIVATIONS[self.activation_type](*self.activation_params.values()))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _n_features(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
       # Make sure to not mess up the random state.
        # try:
            # ====== YOUR CODE: ======

        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            # ====== YOUR CODE: ======
            a, b = self.layer_inputs[-1]
            return a * b * self.channels[-1]
           # ========================
        finally:
            torch.set_rng_state(rng_state)

    def _make_classifier(self):
        layers = []
        # Discover the number of features after the CNN part.
        in_features = self._n_features()

        # TODO: Create the classifier part of the model:
        #  (FC -> ACT)*M -> Linear
        #  The last Linear layer should have an output dim of out_classes.
        # ====== YOUR CODE: ======
        for dim in self.hidden_dims:
            layers.append(nn.Linear(int(in_features), dim))
            in_features = dim
            layers.append(ACTIVATIONS[self.activation_type](*self.activation_params.values()))
        #add final linear layer for classes
        layers.append(nn.Linear(in_features, self.out_classes))
        # ========================

        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        # ====== YOUR CODE: ======
        features = self.feature_extractor(x)
        out = self.classifier(features.view(features.shape[0], -1))
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        batchnorm: bool = False,
        dropout: float = 0.0,
        activation_type: str = "relu",
        activation_params: dict = {},
        **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  - main_path, which should contain the convolution, dropout,
        #    batchnorm, relu sequences (in this order).
        #    Should end with a final conv as in the diagram.
        #  - shortcut_path which should represent the skip-connection and
        #    may contain a 1x1 conv.
        #  Notes:
        #  - Use convolutions which preserve the spatial extent of the input.
        #  - Use bias in the main_path conv layers, and no bias in the skips.
        #  - For simplicity of implementation, assume kernel sizes are odd.
        #  - Don't create layers which you don't use! This will prevent
        #    correct comparison in the test.
        # ====== YOUR CODE: ======

        layers = []
        shortcut_layers = []

        first_in_channels = in_channels
        for i, out_channels in enumerate(channels):
          kernal_size = kernel_sizes[i]
          layers.append(nn.Conv2d(in_channels, out_channels, kernal_size , padding=((kernal_size - 1) // 2), bias=True))
          in_channels = out_channels

          if i+1 == len(channels):
            break
          if dropout > 0.0:
              layers.append(nn.Dropout2d(p = dropout))
          if batchnorm:
              layers.append(nn.BatchNorm2d(out_channels))
          layers.append(ACTIVATIONS[activation_type](*activation_params.values()))

        if first_in_channels != channels[-1]:
          shortcut_layers.append(nn.Conv2d(first_in_channels, channels[-1], kernel_size=1, bias=False)) 
        else:
          shortcut_layers.append(nn.Identity())

        self.main_path = nn.Sequential(*layers)
        self.shortcut_path = nn.Sequential(*shortcut_layers)
     
        # ========================

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        out = torch.relu(out)
        return out


class ResidualBottleneckBlock(ResidualBlock):
    """
    A residual bottleneck block.
    """

    def __init__(
        self,
        in_out_channels: int,
        inner_channels: Sequence[int],
        inner_kernel_sizes: Sequence[int],
        **kwargs,
    ):
        """
        :param in_out_channels: Number of input and output channels of the block.
            The first conv in this block will project from this number, and the
            last conv will project back to this number of channel.
        :param inner_channels: List of number of output channels for each internal
            convolution in the block (i.e. not the outer projections)
            The length determines the number of convolutions.
        :param inner_kernel_sizes: List of kernel sizes (spatial) for the internal
            convolutions in the block. Length should be the same as inner_channels.
            Values should be odd numbers.
        :param kwargs: Any additional arguments supported by ResidualBlock.
        """
        # ====== YOUR CODE: ======
        in_channels = in_out_channels
        channels = [inner_channels[0]] + (inner_channels) + [in_out_channels]
        self.channels = channels
        kernels = [1] + (inner_kernel_sizes) + [1]
        ResidualBlock.__init__(self,
                               in_channels=in_channels,
                               channels=channels,
                               kernel_sizes=kernels,
                               **kwargs)
        # ========================


class ResNetClassifier(ConvClassifier):
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        batchnorm=False,
        dropout=0.0,
        **kwargs,
    ):
        """
        See arguments of ConvClassifier & ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )
        
    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions.
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        # ====== YOUR CODE: ======
        # Loop over groups of P output channels and create a block from them.
        N = len(self.channels)
        P = self.pool_every
        output_channels = [in_channels] + list(self.channels)
        
        self.layer_inputs = []

        new_in_h = in_h
        new_in_w = in_w

        for i in range(0, N, P):           
            if (i+P+1) <= len(output_channels):
              channels_per_block = output_channels[i+1:i+P+1] 
            else:
              channels_per_block = output_channels[i+1:]
            
            in_channels = output_channels[i]
            res_block = ResidualBlock(in_channels,
                                      channels_per_block,
                                      kernel_sizes=[3]*len(channels_per_block),
                                      batchnorm=self.batchnorm,
                                      dropout=self.dropout,
                                      activation_type=self.activation_type,
                                      activation_params=self.activation_params
                                      )
            layers.append(res_block)

            for t in range(P):
              new_in_h = ((new_in_h + 2 * int((3 - 1) / 2) - 1 * (3 - 1) - 1) // (1)) + 1
              new_in_w = ((new_in_w + 2 * int((3 - 1) / 2) - 1 * (3 - 1) - 1) // (1)) + 1
              self.layer_inputs.append((new_in_h, new_in_w))

            if i + P <= N:
              layers.append(POOLINGS[self.pooling_type](*self.pooling_params.values()))
              new_in_h = int((new_in_h - self.pooling_params['kernel_size']) / self.pooling_params['kernel_size']) + 1
              new_in_w = int((new_in_w - self.pooling_params['kernel_size']) / self.pooling_params['kernel_size']) + 1
              self.layer_inputs.append((new_in_h, new_in_w))  
        # ========================
        seq = nn.Sequential(*layers)
        return seq


class YourCodeNet(ConvClassifier):
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        batchnorm=True,
        dropout=0.5,
        conv_params ={'kernel_size':3},
        pooling_params= {'kernel_size':3},
        **kwargs,
    ):
        """
        See ConvClassifier.__init__
        """
        self.batchnorm = batchnorm
        self.dropout = dropout

        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, conv_params=conv_params, pooling_params=pooling_params,**kwargs
        )

        # super().__init__(
        #   in_size,
        #   out_classes,
        #   channels,
        #   pool_every,
        #   hidden_dims,
        #   # batchnorm=True,
        #   # dropout=0.5,
        #   **kwargs,

        # )

        # self.conv_params = conv_params
        # self.pooling_params = pooling_params
        # TODO: Add any additional initialization as needed.
        # ====== YOUR CODE: ======
        # ========================
    def _make_feature_extractor(self):
            in_channels, in_h, in_w, = tuple(self.in_size)

            layers = []
            # TODO: Create the feature extractor part of the model:
            #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
            #   \------- SKIP ------/
            #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions.
            #  Apply Pooling to reduce dimensions after every P convolutions.
            #  Notes:
            #  - If N is not divisible by P, then N mod P additional
            #    CONV->ACT (with a skip over them) should exist at the end,
            #    without a POOL after them.
            #  - Use your own ResidualBlock implementation.
            # ====== YOUR CODE: ======
            # Loop over groups of P output channels and create a block from them.
            N = len(self.channels)
            P = self.pool_every
            output_channels = [in_channels] + list(self.channels)
            
            self.layer_inputs = []

            new_in_h = in_h
            new_in_w = in_w

            for i in range(0, N, P):           
                if (i+P+1) <= len(output_channels):
                  channels_per_block = output_channels[i+1:i+P+1] 
                else:
                  channels_per_block = output_channels[i+1:]
                
                in_channels = output_channels[i]
                res_block = ResidualBlock(in_channels,
                                          channels_per_block,
                                          kernel_sizes=[3]*len(channels_per_block),
                                          batchnorm=self.batchnorm,
                                          dropout=self.dropout,
                                          activation_type=self.activation_type,
                                          activation_params=self.activation_params
                                          )
                layers.append(res_block)

                for t in range(P):
                  new_in_h = ((new_in_h + 2 * int((3 - 1) / 2) - 1 * (3 - 1) - 1) // (1)) + 1
                  new_in_w = ((new_in_w + 2 * int((3 - 1) / 2) - 1 * (3 - 1) - 1) // (1)) + 1
                  self.layer_inputs.append((new_in_h, new_in_w))

                if i + P <= N:
                  layers.append(POOLINGS[self.pooling_type](*self.pooling_params.values()))
                  new_in_h = int((new_in_h - self.pooling_params['kernel_size']) / self.pooling_params['kernel_size']) + 1
                  new_in_w = int((new_in_w - self.pooling_params['kernel_size']) / self.pooling_params['kernel_size']) + 1
                  self.layer_inputs.append((new_in_h, new_in_w))  

            # ========================
            seq = nn.Sequential(*layers)
            return seq
