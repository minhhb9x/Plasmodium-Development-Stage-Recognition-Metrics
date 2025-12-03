import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from utils import box_iou_calc, xywhn2xyxy, load_json_gt_and_txt_det_folders


class DetectionConfusionMatrix:
    def __init__(self, num_classes: int, 
                 class_name: list[str], 
                 gt_class_name: list[str],
                #  mapping_gt_pred: dict,
                 CONF_THRESHOLD=0.3, 
                 IOU_THRESHOLD=0.5):
        
        self.num_classes = num_classes
        self.class_name = class_name
        if len(class_name) != num_classes:
            raise ValueError("Number of class names must be equal to num_classes + 1")

        self.gt_class_name = gt_class_name
        self.num_gt_classes = len(gt_class_name)
        self.matrix = np.zeros((num_classes + 1, self.num_gt_classes + 1), dtype=np.int64)

        # self.mapping_gt_pred = mapping_gt_pred

        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.IOU_THRESHOLD = IOU_THRESHOLD

        # counting number of difficult boxes
        self.num_difficult_boxes = 0
        self.num_correct_ambiguity = 0

    def process_batch(self, detections, labels: np.ndarray):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 6]), class, x1, y1, x2, y2, attributes
        Returns:
            None, updates confusion matrix accordingly
        """
        gt_classes = labels[:, 0].astype(np.int16)

        try:
            detections = detections[detections[:, 4] > self.CONF_THRESHOLD]
        except (IndexError, TypeError): # If detections is empty or not a valid array
            for i, label in enumerate(labels):
                gt_class = gt_classes[i]
                self.matrix[self.num_classes, gt_class] += 1  # FP (not detected)
            return

        detection_classes = detections[:, 5].astype(np.int16)
        all_ious = box_iou_calc(labels[:, 1:5], detections[:, :4])
        want_idx = np.where(all_ious > self.IOU_THRESHOLD)

        all_matches = [[want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]]
                       for i in range(want_idx[0].shape[0])]
        all_matches = np.array(all_matches)

        if all_matches.shape[0] > 0:
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]
            all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]
            all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]

        # Process matches
        for i, label in enumerate(labels):
            gt_class = gt_classes[i]
            match = all_matches[all_matches[:, 0] == i]
            if match.shape[0] == 1:
                det_idx = int(match[0, 1])
                pred_class = detection_classes[det_idx]
                self.matrix[pred_class, gt_class] += 1  # TP
                if label[0] == 9:
                    self.num_correct_ambiguity += self.check_correct_ambiguity(pred_class, label[5])
            else:
                self.matrix[self.num_classes, gt_class] += 1  # FN

        for i, detection in enumerate(detections):
            match = all_matches[all_matches[:, 1] == i]
            if match.shape[0] == 0:
                pred_class = detection_classes[i]
                self.matrix[pred_class, self.num_gt_classes] += 1  # FP

    def check_correct_ambiguity(self, pred_class, ambiguity_list):
        mapping_index_attributes = {
            "Ambiguity with R": 0,
            "Ambiguity with T": 1,
            "Ambiguity with S1": 2,
            "Ambiguity with S2": 3,
            "Ambiguity with G1": 4,
            "Ambiguity with G2-5": 5,
            "Ambiguity with Unp": 6,
            "Ambiguity with UnpArt": 7,
            "Ambiguity with UnpDK": 8,
        }
        ambiguity_matrix = np.zeros((5, 9), dtype=int)
        ambiguity_matrix[0, 0] = 1  # R-R
        ambiguity_matrix[1, 1] = 1  # T-T
        ambiguity_matrix[2, 2] = 1  # S-S1
        ambiguity_matrix[2, 3] = 1  # S-S2
        ambiguity_matrix[3, 4] = 1  # G-G1
        ambiguity_matrix[3, 5] = 1  # G-G2-5
        ambiguity_matrix[4, 6] = 1  # Uninf-Unp
        ambiguity_matrix[4, 7] = 1  # Uninf-UnpArt
        ambiguity_matrix[4, 8] = 1  # Uninf-UnpDK
        cnt = 0
        res = 0
        for attr in ambiguity_list:
            if attr['value'] == 'true':
                cnt = 1
                gt_class = mapping_index_attributes.get(attr['name'], -1)
                if ambiguity_matrix[pred_class, gt_class] == 1:
                    res = 1
                    break
        self.num_difficult_boxes += cnt
        return res

    def return_matrix(self):
        return self.matrix

    def print_matrix(self):
        print("Confusion Matrix (rows = predicted, cols = ground truth):")
        for i in range(self.num_classes + 1):
            print(' '.join(map(str, self.matrix[i])))

    def compute_PR_from_matrix(self, confusion_matrix):
        precision_recall_metrics = {}
        for i in range(self.num_classes):
            tp = confusion_matrix[i, i]
            fp = sum(confusion_matrix[i, :]) - tp
            fn = sum(confusion_matrix[:, i]) - tp

            precision = tp / (tp + fp) if (tp + fp) != 0 else 0
            recall = tp / (tp + fn) if (tp + fn) != 0 else 0

            precision_recall_metrics[f'precision/{self.class_name[i]}'] = float(precision)
            precision_recall_metrics[f'recall/{self.class_name[i]}'] = float(recall)

        # Tổng hợp parasitized (ví dụ từ class 0 đến 3)
        parasitized_correct = 0
        parasitized_recall = 0
        parasitized_precision = 0

        for i in range(4):  # Giả sử parasitized là class 0–3
            parasitized_correct += confusion_matrix[i, i]
            parasitized_recall += sum(confusion_matrix[:, i])
            parasitized_precision += sum(confusion_matrix[i, :])

        precision_recall_metrics['precision/parasitized'] = (
            parasitized_correct / parasitized_precision if parasitized_precision != 0 else 0
        )
        precision_recall_metrics['recall/parasitized'] = (
            parasitized_correct / parasitized_recall if parasitized_recall != 0 else 0
        )

        return precision_recall_metrics
    
    def plot_matrix(self, save_path="confusion_matrix.png", normalize=False, cmap="Blues"):
        """
        Plot and save the confusion matrix as an image (rows: predicted, columns: ground truth).
        
        Args:
            save_path (str): Path to save the image file.
            normalize (bool): If True, normalize the matrix by row to show percentages.
            cmap (str): Colormap for the plot.
        """

        import seaborn  

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(15, 9), tight_layout=True)
        nc, label_nn = self.num_classes, len(self.class_name)  # number of classes, names
        seaborn.set_theme(font_scale=2.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < label_nn < 99) and (label_nn == nc)  # apply names to ticklabels
        yticklabels = list(self.class_name + ['background']) if labels else "auto"
        xticklabels = list(self.gt_class_name + ['background']) if labels else "auto"
        
       
        seaborn.heatmap(
                array,
                ax=ax,
                annot=nc < 30,
                annot_kws={"size": 30},
                cmap=cmap,
                fmt=".2f" if normalize else ".0f",
                square=False,
                vmin=0.0,
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                cbar=False,
                norm=LogNorm(),
            ).set_facecolor((1, 1, 1))
        title = "Confusion Matrix" + " Normalized" * normalize
        ax.set_xlabel("True", fontsize=20)
        ax.set_ylabel("Predicted", fontsize=20)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)
        # ax.set_title(title)
        plot_fname = save_path
        os.makedirs(os.path.dirname(plot_fname), exist_ok=True)
        fig.savefig(plot_fname, dpi=500, bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    W, H = 3072, 2048
    class_name = ["R", "T", "S", "G", "Uninf"]
    gt_class_name = ["R", "T", "S1", "S2", "G1", "G2-5", 
                     "Unp", "UnpArt", "UnpDK", "D"]

    pred_gt_matching = {pred: {gt: 0 for gt in gt_class_name} for pred in class_name}

    # Gán giá trị
    pred_gt_matching["R"]["R"] = 1
    pred_gt_matching["T"]["T"] = 1
    pred_gt_matching["S"]["S1"] = 1
    pred_gt_matching["S"]["S2"] = 1
    pred_gt_matching["G"]["G1"] = 1
    pred_gt_matching["G"]["G2-5"] = 1
    pred_gt_matching["Uninf"]["Unp"] = 1
    pred_gt_matching["Uninf"]["UnpArt"] = 1
    pred_gt_matching["Uninf"]["UnpDK"] = 1

    mapping_gt_index = {
                    "R": 0,
                    "T": 1,
                    "S1": 2,
                    "S2": 3,
                    "G1": 4,
                    "G2-5": 5,
                    "Unp": 6,
                    "UnpArt": 7,
                    "UnpDK": 8,
                    "D": 9
                    }
    
    cm = DetectionConfusionMatrix(num_classes=len(class_name), 
                                  class_name=class_name, 
                                  gt_class_name=gt_class_name,
                                  CONF_THRESHOLD=0.25, 
                                  IOU_THRESHOLD=0.45)
    
    args ={
        'det_dir': 'txt_output/v11s_coco_v2data_PA7_5_classes_500ep_agnostic_conf=0.25',
        'gt_dir': 'Json_tif_malaria/test/labels',
        'confusion_matrix': 'confusion_matrix/our_json_data/' + 'rectangle_' + \
              'v11s_coco_v2data_PA7_5_classes_500ep_agnostic_conf=0.25' + '.jpg',

        'gt_index': mapping_gt_index,
        # 'confusion_matrix': 'draft.png'
    }

    gt_det_dict =  load_json_gt_and_txt_det_folders(args['gt_dir'], args['det_dir'], args['gt_index'], W, H)
    
    for image, boxes in gt_det_dict.items():
        gt, det = boxes
        cm.process_batch(det, gt)

    print("Number of difficult boxes:", cm.num_difficult_boxes)
    print("Number of correctly predicted difficult boxes:", cm.num_correct_ambiguity)
    print("Accuracy on difficult boxes: {:.2f}%".format(
        100 * cm.num_correct_ambiguity / cm.num_difficult_boxes if cm.num_difficult_boxes > 0 else 0.0
    ))
    cm.print_matrix()
    cm.plot_matrix(save_path=args['confusion_matrix'], normalize=False, cmap="Reds")