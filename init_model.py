
import torch
import torchvision.transforms as transforms
from PIL import Image

PATH_TO_MODEL = './models/model-best4_best.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
imsize = 256
loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])


def loadModel(path):
    return torch.load(path, map_location=device)


def get_loaded_model():
    model = loadModel(PATH_TO_MODEL)
    model = model.to(device)
    model.eval()
    return model


def fix_image(image):
    image = Image.fromarray(image)
    image = loader(image).float()
    image = torch.autograd.Variable(image, requires_grad=True)
    image = image.view(1, 3, imsize, imsize)
    image = image.to(device)
    return image


def get_label(code):
    code = torch.argmax(code).item()
    if code == 1:
        return 'WithoutMask'
    else:
        return 'WithMask'
