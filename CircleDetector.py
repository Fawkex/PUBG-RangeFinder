import torch
from torch.autograd import Variable

from torchvision.models.resnet import conv3x3
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)                # 原地替换 节省内存开销
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample                     # shortcut
        
    def forward(self, x):
        residual=x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if(self.downsample):
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# 自定义一个神经网络，使用nn.model，，通过__init__初始化每一层神经网络。
# 使用forward连接数据
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layers(block, 16, layers[0])
        self.layer2 = self._make_layers(block, 32, layers[1], 2)
        self.layer3 = self._make_layers(block, 64, layers[2], 2)
        self.layer4 = self._make_layers(block, 128, layers[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    # _make_layers函数重复残差块，以及shortcut部分
    def _make_layers(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):      # 卷积核为1 进行升降维
            downsample = nn.Sequential(                        # stride==2的时候 也就是每次输出信道升维的时候
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


from torchvision.transforms import transforms
import onnxruntime as rt
import numpy as np

class CircleDetector:

    classes =  ['player', 'mark', 'waypoint', 'signal_tower', 'others']

    transformations = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    def __init__(self, model_path, model = None, is_onnx = False):
        number_of_labels = 5
        self.is_onnx = is_onnx
        if is_onnx:
            if not model:
                session_options = rt.SessionOptions()
                session_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.model = rt.InferenceSession(
                    model_path,
                    providers=rt.get_available_providers(),
                    sess_options=session_options)
            else:
                self.model = model
            self.input_name = self.model.get_inputs()[0].name
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if not model:
                self.model = ResNet(ResidualBlock, [3,3,3,3], number_of_labels).to(device='cuda:0')
                self.model.load_state_dict(torch.load(model_path))
            else:
                self.model = model
            self.model.to(self.device)
            self.model.eval()

    def infer_circles(self, images):
        if len(images) == 0:
            return []
        if self.is_onnx:
            images = np.array([ self.transformations(x).numpy() for x in images ])
            outputs = self.model.run(None, {self.input_name: images})[0]
            predicted = np.argmax(outputs, axis=1)
        else:
            images = torch.stack([ self.transformations(x) for x in images ])
            images = Variable(images.to(self.device))
            outputs = self.model(images)
            _, predicted = torch.max(outputs, 1)
        predicted_class = [ self.classes[x] for x in predicted ]
        return predicted_class