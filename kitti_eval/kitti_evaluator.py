import os
import abc
import copy

from .utils.dist_utils import all_gather
from .utils.eval_utils import kitti_eval

class KittiEvaluator(object):
    __metaclass__ = abc.ABCMeta
    def __init__(
        self, 
        gt_anns,
        current_classes = ['Car', 'Pedestrian', 'Cyclist'],
        eval_types=['bbox', '3d', 'bev'],
    ):
        # expects GT to be a list of dictionaries with the required fields 
        # as in the KITTI dataset (name, truncated... etc.)
        self.gt_anns = copy.deepcopy(gt_anns)
        self.dt_annos = []

        self.current_classes = current_classes
        self.eval_types = eval_types

    def update(self, predictions):
        """update dt_annos with the new predictions.
        """
        self.dt_annos.append(predictions)

    def synchronize_between_processes(self,):
        """Gather results from multiple GPUs. (Does not have any effect with single GPU)
        """
        all_dt_annos = all_gather(self.dt_annos)
        self.dt_annos = []

        for ann in all_dt_annos:
            self.dt_annos.extend(ann)
    
    @abc.abstractmethod
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
        return self.dt_annos
        
    def evaluate(self, submission_prefix=None):
        """Evaluate the accumulated results.

            Optionally save the results in submission format.
        """

        results = self.format_results()

        ap_result_str, ap_dict = kitti_eval(
            self.gt_anns, 
            results,
            self.current_classes, 
            self.eval_types)
        
        # save file in submission format
        if submission_prefix is not None:
            os.makedirs(submission_prefix, exist_ok=True)

            print(f'Saving KITTI submission to {submission_prefix}')
            for i, anno in enumerate(self.dt_annos):
                sample_idx = self.anno_infos[i]['image']['image_idx']
                cur_det_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(cur_det_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions'][::-1]  # lhw -> hwl
                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} '
                            '{:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f}'.format(
                                anno['name'][idx],
                                anno['alpha'][idx],
                                *bbox[idx],  # 4 float
                                *dims[idx],  # 3 float
                                *loc[idx],  # 3 float
                                anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f,
                        )
                print(f'Result is saved to {submission_prefix}')
        
        return ap_result_str, ap_dict