
import matplotlib.pyplot as plt
# from util.plot_utils import plot_logs, plot_precision_recall
from pathlib import Path, PurePath
import pandas as pd 

# log_path = [Path("checkpoint/TL_kitti")]
# log_name = 'log.txt'
# log_file = "checkpoint/TL_kitti/log.txt"
# log_file1 = "checkpoint/log.txt"

log_file_names = ['deformable_iterative_proposals']
log_files = ["/raid/data/ocalikka/deformable-detr/exps/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage/log.txt"]
dfs = [pd.read_json(log_file, lines=True) for log_file in log_files]

fields = dfs[0].keys()
x_axis = "epoch"

for field in fields:
    plt.figure()
    for df in dfs:
        plt.plot(df[x_axis], df[field], linewidth=2)
    plt.title(field)
    plt.legend(log_file_names)
    plt.xlabel(x_axis)
    plt.ylabel(field)
    plt.grid()

    plt.tight_layout()
    plt.savefig('/raid/data/ocalikka/deformable-detr/exps/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage/plots/'+field+'.png')
    
    # plt.show()
# df = pd.read_json(log_file, lines=True)
# df1 = pd.read_json(log_file1, lines=True)
print('done')
# dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]
# plot_logs(log_file)
# plot_precision_recall(log_file+'/log.txt')