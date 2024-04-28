import timm
import torch
from timm.models import create_model
import models
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
import argparse
from PIL import Image
from torch.autograd import Variable
import cv2
import os
import numpy as np

parser = argparse.ArgumentParser(description='IA-RED^2 Interpretation Tool')
parser.add_argument('-p', '--img-path', metavar='PATH',
                    help='path to input image')
parser.add_argument('-o', '--output', metavar='DIR', default='./output/',
                    help='path to the output dir')

def tensor2im_norm(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.detach().cpu().float().numpy()
    image_numpy = image_numpy - np.min(image_numpy)
    image_numpy = image_numpy / np.max(image_numpy) * 255
    image_numpy = np.maximum(image_numpy, 0)
    image_numpy = np.minimum(image_numpy, 255)
    return image_numpy.astype(imtype)


def main(args):
    
    device = 'cpu'#torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    
    print(f"Creating model: interp_deit_small_patch16_224")
#     model = create_model(
#         'interp_deit_small_patch16_224',
#         pretrained=True,
#         num_classes=1000,
#     )
    model = models.interp_deit_base_patch16_224(device, pretrained=True)
    checkpoint = torch.load('./ckpt.pth', map_location='cpu')
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()

    t = []
    t.append(
        transforms.Resize((224, 224), interpolation=3),
    )
    t.append(transforms.ToTensor())
    transform = transforms.Compose(t)
    norm = transforms.Compose([transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
    image = Image.open(args.img_path)
    image.convert('RGB')
    image = transform(image)
    img_tensor = norm(image).to(device)
    img_variable = Variable(img_tensor.unsqueeze(0))
    
    with torch.no_grad():
        preds, _, soft_policys = model(img_variable)
        print(preds.argmax())
#     preds, flops, probs

    soft_map = soft_policys[1][:,1:]
    hm = soft_map.view(soft_map.size(0), 14, 14)
    hm = hm.squeeze()
    hm = torch.clamp(hm, 0.48, 1)
    
    
    vis = tensor2im_norm(hm.view(14, 14, 1))

    
    np.savez('cam', **{'cam': vis})
    
    h, w = image.size(1), image.size(2)
    heatmap = cv2.applyColorMap(cv2.resize(vis,(w, h)), cv2.COLORMAP_JET)
    image = image * 255
    original = image.transpose(0,1).transpose(1,2).cpu().numpy().astype(np.uint8)
    original = original[:,:,::-1]
    heatmap = heatmap.astype(np.uint8)
    result = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    cv2.imwrite(os.path.join(args.output, 'interpret-result.png'), result)
    print('Results are saved at {}'.format(os.path.join(args.output, 'interpret-result.png')))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)