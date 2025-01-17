"""Minimal script to load and run the model on an arbitrary image.
"""
import argparse
from torch._C import device
from models import build_model
from PIL import Image
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False)

from main import get_args_parser


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device='cuda')
    return b

def detect(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0).to(device='cuda')

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    # print(10*'*', 'image shape -2: ', img.shape[-2])
    # print(10*'*', 'image shape -1: ', img.shape[-1])

    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled


def plot_results(pil_img, prob, boxes, img_id, plot_save_dir='./saved_plots'):

    os.makedirs(plot_save_dir, exist_ok=True)

    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_save_dir, img_id+'.png'))
    print(f'{img_id} is saved under {plot_save_dir}')
    # plt.show()


if __name__ == '__main__':

    # Kitti dataset path
    kitti_dir = '/mnt/lustre/public_datasets/kitti'
    # Trained model checkpoint path
    model_path = 'exps/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage/checkpoint0049.pth'

    parser = argparse.ArgumentParser('Deformable DETR test script', parents=[get_args_parser()])
    args = parser.parse_args()
    detr, criterion, postprocessors = build_model(args)
    state_dict = torch.load(model_path)
    detr.load_state_dict(state_dict['model'])
    detr.eval().to(device='cuda')

    # KITTI classes
    CLASSES = [
        'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram'
    ]

    # Colors are for visualization
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
            [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    # Standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize(400),  # kitti minimum image side is the height of 400 px
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Select the test image id for inference
    # Note that you can provide any arbitrary image path
    kitti_img_id = 200
    img_id = '{0:06d}'.format(kitti_img_id)
    img_path = os.path.join(kitti_dir, 'testing/image_2/'+img_id+'.png')
    # Load the image
    im = Image.open(img_path)
    # Run inference
    scores, boxes = detect(im, detr, transform)

    ## Uncomment to save plots
    # plot_save_dir = '/raid/data/ocalikka/deformable-detr/exps/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage/output_frames/'
    # plot_results(im, scores, boxes, img_id=img_id, plot_save_dir=plot_save_dir)
