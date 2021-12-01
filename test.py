"""Run test on kitti test split.
"""
from inference import *  # helper funcs
import numpy as np
import pandas as pd
from kitti_eval.kitti_evaluator import KittiEvaluator
from kitti_eval.utils import data_utils
from tqdm import tqdm


kitti_dir = '/mnt/lustre/public_datasets/kitti'
model_path = 'exps/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage/checkpoint0049.pth'

gt_label_path = os.path.join(kitti_dir, 'training/label_2/')
image_path = os.path.join(kitti_dir, 'training/image_2/')
splits_path = os.path.join(kitti_dir, 'training/splits')

split_file = os.path.join(splits_path, 'test_split.txt')
assert os.path.isfile(split_file), f"Split file at {split_file} is not found!"

# Read the label ids from the test split
ids = list()
with open(split_file) as file:
    for line in file:
        ids.append(line.rstrip())


class_mapping = {
    0:'Car', 1:'Van', 2:'Truck', 3:'Pedestrian',
    4:'Person_sitting', 5:'Cyclist', 6:'Tram'
}

parser = argparse.ArgumentParser('DETR training and evaluation script',
                                parents=[get_args_parser()])
args = parser.parse_args()
detr, criterion, postprocessors = build_model(args)
state_dict = torch.load(model_path)
detr.load_state_dict(state_dict['model'])
detr.eval().to(device='cuda')

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(400),  # kitti minimum side is height of 400 px
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

kitti_keys = ['name', 'truncated', 'occluded', 'alpha', 'bbox',
                'dimensions', 'location', 'rotation_y', 'score']
gt_annos = []
pred_annos = []

print(f'\nInference started on the test set with {len(ids)} samples..\n')

for img_id in tqdm(ids):

    img_path = image_path+img_id+'.png'
    lbl_path = gt_label_path+img_id+'.txt'

    im = Image.open(img_path)

    scores, boxes = detect(im, detr, transform)

    ann_data = pd.read_csv(lbl_path, sep=" ", names=['label', 'truncated', 'occluded',
                                                    'alpha', 'bbox_xmin', 'bbox_ymin',
                                                    'bbox_xmax', 'bbox_ymax', 'dim_height',
                                                    'dim_width', 'dim_length', 'loc_x',
                                                    'loc_y', 'loc_z', 'rotation_y', 'score'])

    ann_data = ann_data[ann_data['label']!='DontCare']  # Remove DontCare
    kitti_gt_ann = {k: [] for k in kitti_keys}
    kitti_pred = {k: [] for k in kitti_keys}
    kitti_gt_ann['name'] = ann_data['label'].to_numpy(dtype=np.str_, copy=True)
    kitti_gt_ann['truncated'] = ann_data['truncated'].to_numpy(copy=True)
    kitti_gt_ann['occluded'] = ann_data['occluded'].to_numpy(copy=True)
    kitti_gt_ann['alpha'] = ann_data['alpha'].to_numpy(copy=True)
    kitti_gt_ann['bbox'] = ann_data[['bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax']].to_numpy(copy=True)
    kitti_gt_ann['dimensions'] = ann_data[['dim_height', 'dim_width', 'dim_length']].to_numpy(copy=True)
    kitti_gt_ann['location'] = ann_data[['loc_x', 'loc_y', 'loc_z']].to_numpy(copy=True)
    kitti_gt_ann['rotation_y'] = ann_data['rotation_y'].to_numpy(copy=True)
    kitti_gt_ann['score'] = np.array([0. for n in ann_data['score']])
    gt_annos.append(kitti_gt_ann)

    kitti_pred['name'] = np.array([class_mapping[torch.argmax(n).item()] for n in scores])
    kitti_pred['bbox'] = np.array([n.cpu().numpy() for n in boxes])
    kitti_pred['score'] = np.array([torch.max(n).item() for n in scores])
    kitti_pred['alpha'] = np.array([0 for i in range(len(scores))])
    pred_annos.append(kitti_pred)

print('\nCalculating the metric results...\n')

# read the test labels into a dictionary
gt_annos = data_utils.get_label_annos(gt_label_path, image_ids=ids)

# Keep Car, Pedestrian, and Cyclist as the classes for validation and test
# It needs minimum of 500 samples for a correct kitti evaluation
kitti_evaluator = KittiEvaluator(gt_anns=gt_annos,
                                current_classes=['Car', 'Pedestrian', 'Cyclist'],
                                eval_types=['bbox'])


kitti_evaluator.dt_annos = pred_annos
# gather results from multiple GPUs (no effect with single GPU)
kitti_evaluator.synchronize_between_processes()
# evaluate the predictions and get AP results.
ap_str, ap_dict = kitti_evaluator.evaluate()
print(ap_str)
