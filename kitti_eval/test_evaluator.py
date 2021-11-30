import torch
import numpy as np

from kitti_evaluator import KittiEvaluator

# Implementation of a KITTI evaluator using the base class.
# Only <format_results> needs to be implemented.
class MyKittiEvaluator(KittiEvaluator):
    
    def __init__(self, gt_anns, current_classes, eval_types):
        super().__init__(gt_anns, current_classes=current_classes, eval_types=eval_types)
    
    def format_results(self,):
        """Formatter to convert the model outputs to the required form for KITTI evaluator.

            GT annotations and results should have the same form,
            which is a list of dictionaries with the following fields
            (where each element in the list corresponds to an image):
            {
                 'name': <np.ndarray> (n),
                 'truncated': <np.ndarray> (n),
                 'occluded': <np.ndarray> (n),
                 'alpha': <np.ndarray> (n),
                 'bbox': <np.ndarray> (n, 4),
                 'dimensions': <np.ndarray> (n, 3),
                 'location': <np.ndarray> (n, 3),
                 'rotation_y': <np.ndarray> (n),
                 'score': <np.ndarray> (n),

            }

            Args:
                None

            Returns:
                list[dict]: a list of dictionaries complying to KITTI format
        """
        
        # for this test, detections are already in correct form
        return self.dt_annos


def test_evaluator_LSVM():
    from utils import data_utils
    if not torch.cuda.is_available():
        raise RuntimeError('test requires GPU and torch+cuda')
    
    print('Processing the test data...')
    # load LSVM baseline detections (training set)
    #     http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d
    gt_label_path = 'test_data/gt_labels'
    result_path = 'test_data/det_lsvm'
    # read the test labels into a dictionary
    gt_annos = data_utils.get_label_annos(gt_label_path)
    # read the test detections into a dictionary
    predictions = data_utils.get_label_annos(result_path)

    print('Evaluating the test results...')
    # Instantiate our evaluator for 'Car' class to be evaluated with 2d bbox
    kitti_evaluator = MyKittiEvaluator(gt_annos, current_classes=['Car'], eval_types=['bbox'])

    # update the evaluator with the results from each batch.
    for pred in predictions:
        kitti_evaluator.update(pred)
        # use similary, when inserted in a dataloader loop
        # output = model(input_batch)
        # kitti_evaluator.update(output)

    # gather results from multiple GPUs (no effect with single GPU)
    kitti_evaluator.synchronize_between_processes()
    # evaluate the predictions and get AP results.
    ap_str, ap_dict = kitti_evaluator.evaluate()
    print(ap_str)
    # reference values obtained using the object development kit:
    #     http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d
    assert np.isclose(ap_dict['Car_bbox_easy'], 74.73071304)
    assert np.isclose(ap_dict['Car_bbox_moderate'], 58.92306159)
    assert np.isclose(ap_dict['Car_bbox_hard'], 49.38764498)
   

def test_evaluator_noisy():
    from utils import data_utils
    if not torch.cuda.is_available():
        raise RuntimeError('test requires GPU and torch+cuda')
    
    print('Processing the test data...')
    # load noisy labels as detections (training set)
    gt_label_path = 'test_data/gt_labels'
    result_path = 'test_data/det_noisy'
    # read the test labels into a dictionary
    gt_annos = data_utils.get_label_annos(gt_label_path)
    # read the test detections into a dictionary
    predictions = data_utils.get_label_annos(result_path)

    print('Evaluating the test results...')
    # Instantiate our evaluator for 'Car' class to be evaluated in 2d (bbox), 3d and BEV performance.
    curr_classes = ['Car', 'Pedestrian']
    eval_types = ['bbox', '3d', 'bev']
    kitti_evaluator = MyKittiEvaluator(gt_annos, current_classes=curr_classes, eval_types=eval_types)

    # update the evaluator with the results from each batch/image.
    for pred in predictions:
        kitti_evaluator.update(pred)
        # use similary, when inserted in a dataloader loop
        # output = model(input_batch)
        # kitti_evaluator.update(output)

    # gather results from multiple GPUs (no effect with single GPU)
    kitti_evaluator.synchronize_between_processes()
    # evaluate the predictions and get AP results.
    ap_str, ap_dict = kitti_evaluator.evaluate()
    print(ap_str)
    
    # reference values obtained using the object development kit:
    #     http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
    eval_types.append('aos')
    with open('test_data/noisy_reference_AP.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            ret = line.strip().split()
            cls, ev_type, easy_AP, mod_AP, hard_AP = ret
            base_key = '{}_{}'.format(cls, ev_type)
            assert np.isclose(
                ap_dict[base_key + '_easy'], float(easy_AP), rtol=1e-4)
            assert np.isclose(
                ap_dict[base_key + '_moderate'], float(mod_AP), rtol=1e-4)
            assert np.isclose(
                ap_dict[base_key + '_hard'], float(hard_AP), rtol=1e-4)
    

if __name__ == "__main__":
    test_evaluator_LSVM()
    test_evaluator_noisy()
