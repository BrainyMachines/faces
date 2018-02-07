"""Extract deep features using ResNet network model."""
import torchvision.models as models

class ResNetFeatureExtractor(models.ResNet):
    """Extract deep features using ResNet network model."""

    def __init__(self, model):
        block = type(model.layer1[0])
        layers = [len(model.layer1), len(model.layer2), len(model.layer3), len(model.layer4)]
        super(ResNetFeatureExtractor, self).__init__(block, layers)
        self.inplanes = model.inplanes
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
