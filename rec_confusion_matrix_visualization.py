import os
import numpy as np
from utils import load_gt_det_img_folders, box_iou_calc, plot_boxes_with_labels
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

CLASS_NAMES = ['Ring', 'Trophozoite', 'Schizont', 'Gametocyte', 
               'HealthyRBC', 'Other', 'Difficult', 'Background',
               'Noname', 'Noname', 'Noname', 'Noname']




class DetectionConfusionMatrix:
    def __init__(self, num_classes: int, 
                 class_name: list[str], 
                 gt_class_name: list[str],
                 pred_gt_matching: dict,
                 CONF_THRESHOLD=0.3, 
                 IOU_THRESHOLD=0.5,
                 ):
        
        self.num_classes = num_classes
        self.class_name = class_name
        if len(class_name) != num_classes:
            raise ValueError("Number of class names must be equal to num_classes + 1")

        self.gt_class_name = gt_class_name
        self.num_gt_classes = len(gt_class_name)
        self.matrix = np.zeros((num_classes + 1, self.num_gt_classes + 1), dtype=np.int64)

        self.pred_gt_matching = pred_gt_matching

        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.IOU_THRESHOLD = IOU_THRESHOLD

    def process_batch(self, detections, labels, img, img_name, save_dir):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2,
            img (Array[H, W, C]), the input image
            save_dir (str), directory to save plot images
        Returns:
            None, updates confusion matrix accordingly
        """
        gt_classes = labels[:, 0].astype(np.int16)
        os.makedirs(save_dir, exist_ok=True)

        false_det_dir = os.path.join(save_dir, os.path.splitext(img_name)[0]+'_FP.jpg')
        false_gt_dir = os.path.join(save_dir, os.path.splitext(img_name)[0]+'_FN.jpg')
        all_det_dir = os.path.join(save_dir, os.path.splitext(img_name)[0]+'_All.jpg')

        all_det = []  # All detections
        false_det = []  # False detections
        false_gt = []  # Misdetected ground truths

        all_det_labels = []  # Labels for all detections
        false_det_labels = []  # Labels for false detections
        false_gt_labels = []  # Labels for misdetected ground truths

        try:
            detections = detections[detections[:, 4] > self.CONF_THRESHOLD]
        except (IndexError, TypeError): # If detections is empty or not a valid array
            for i, label in enumerate(labels):
                gt_class = gt_classes[i]
                self.matrix[self.num_classes, gt_class] += 1  # FP (not detected)
                false_gt.append(labels[i, :])
                false_gt_labels.append(self.gt_class_name[gt_class])
            plot_boxes_with_labels(np.array(false_gt), false_gt_labels, 
                                   img.copy(), false_gt_dir, )
            return

        detection_classes = detections[:, 5].astype(np.int16)
        all_ious = box_iou_calc(labels[:, 1:], detections[:, :4])
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
            gt_class_name = self.gt_class_name[gt_class]
            if match.shape[0] == 1:
                det_idx = int(match[0, 1])
                det_i = np.concatenate(([detections[det_idx, 5]], detections[det_idx, :4])) # class, x1, y1, x2, y2
                pred_class = detection_classes[det_idx]
                self.matrix[pred_class, gt_class] += 1  # TP

                # Check if the predicted class matches the ground truth class based on pred_gt_matching
                pred_class_name = self.class_name[pred_class]
                if self.pred_gt_matching[pred_class_name][gt_class_name] == 0:
                    pred_gt_class_name = f"{pred_class_name} | GT: {gt_class_name}"

                    false_det.append(det_i) # FP
                    false_det_labels.append(pred_gt_class_name) # FP label
                    false_gt.append(labels[i, :]) # FN
                    false_gt_labels.append(gt_class_name) # FN label

                all_det.append(det_i) # All
                all_det_labels.append(pred_class_name + f" {detections[det_idx, 4]:.2f}") # All label
                    
            else:
                self.matrix[self.num_classes, gt_class] += 1  # FN
                false_gt.append(labels[i, :])
                false_gt_labels.append(gt_class_name)


        for i, detection in enumerate(detections):
            match = all_matches[all_matches[:, 1] == i]
            if match.shape[0] == 0:
                pred_class = detection_classes[i]
                pred_class_name = self.class_name[pred_class]
                self.matrix[pred_class, self.num_gt_classes] += 1  # FP

                det_i = np.concatenate(([detections[i, 5]], detections[i, :4])) # class, x1, y1, x2, y2

                pred_gt_class_name = f"{pred_class_name} | GT: background"
                false_det.append(det_i) # FP
                false_det_labels.append(pred_gt_class_name) # FP label

                all_det.append(det_i) # All
                all_det_labels.append(pred_class_name + f" {detections[i, 4]:.2f}") # All label

        
        plot_boxes_with_labels(np.array(false_det), false_det_labels, 
                               img.copy(), false_det_dir, )
        plot_boxes_with_labels(np.array(false_gt), false_gt_labels, 
                               img.copy(), false_gt_dir, )
        plot_boxes_with_labels(np.array(all_det), all_det_labels, 
                               img.copy(), all_det_dir, )

    def return_matrix(self):
        return self.matrix

    def print_matrix(self):
        print("Confusion Matrix (rows = predicted, cols = ground truth):")
        for i in range(self.num_classes + 1):
            print(' '.join(map(str, self.matrix[i])))

    def compute_PR_from_matrix(self, cm):
        precision_recall = {}

        per_class_precision = []
        per_class_recall = []
        gt_counts = []

        total_samples = cm.sum()

        # ===== Precision / Recall cho từng class =====
        for i in range(self.num_classes):
            tp = cm[i, i]
            fp = cm[i, :].sum() - tp
            fn = cm[:, i].sum() - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            precision_recall[f'precision/{self.class_name[i]}'] = precision
            precision_recall[f'recall/{self.class_name[i]}'] = recall

            per_class_precision.append(precision)
            per_class_recall.append(recall)
            gt_counts.append(cm[:, i].sum())

        # ===== Precision / Recall cho nhóm PARASITIZED (0–3) =====
        parasitized = list(range(4))

        precision_par = sum(
            precision_recall[f'precision/{self.class_name[i]}']
            for i in parasitized
        ) / len(parasitized)

        recall_par = sum(
            precision_recall[f'recall/{self.class_name[i]}']
            for i in parasitized
        ) / len(parasitized)

        precision_recall['precision/parasitized'] = precision_par
        precision_recall['recall/parasitized'] = recall_par

        # ===== Weighted precision / recall (chỉ 4 lớp bệnh) =====
        gt_counts_par = [gt_counts[i] for i in parasitized]
        total_gt_par = sum(gt_counts_par)

        # trọng số = GT_i / tổng GT nhóm parasitized
        weighted_precision_par = sum(
            precision_recall[f'precision/{self.class_name[i]}'] * (gt_counts[i] / total_gt_par)
            for i in parasitized
        )

        weighted_recall_par = sum(
            precision_recall[f'recall/{self.class_name[i]}'] * (gt_counts[i] / total_gt_par)
            for i in parasitized
        )

        precision_recall['weighted/precision_parasitized'] = weighted_precision_par
        precision_recall['weighted/recall_parasitized'] = weighted_recall_par

        # ===== In kết quả =====
        print("\n===== PRECISION & RECALL =====")
        for k, v in precision_recall.items():
            print(f"{k:35s}: {v:.4f}")
        print("================================\n")

        return precision_recall
    

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
        fig.savefig(plot_fname, dpi=500, bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    # W, H = 3072, 2048
    # Our data
    class_name = ["R", "T", "S", "G", "Uninf"]
    gt_class_name = ["R", "T", "S1", "S2", "G1", "G2-5", 
                     "Unp", "UnpArt", "UnpDK", "D"]

    # # IML
    # class_name = ["R", "T", "S", "G", "Uninf"]
    # gt_class_name = ["R", "T", "S", "G", "Uninf", "Other", "D"]

    # # bbbc041
    # class_name = ["R", "T", "S", "G", "Uninf", "Leu"]
    # gt_class_name = ["R", "T", "S", "G", "Uninf", "Leu", "D"]

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

    # pred_gt_matching["R"]["R"] = 1
    # pred_gt_matching["T"]["T"] = 1
    # pred_gt_matching["S"]["S"] = 1
    # pred_gt_matching["G"]["G"] = 1
    # pred_gt_matching["Uninf"]["Uninf"] = 1
    # pred_gt_matching["Leu"]["Leu"] = 1
    

    cm = DetectionConfusionMatrix(num_classes=len(class_name), 
                                  class_name=class_name, 
                                  gt_class_name=gt_class_name,
                                  CONF_THRESHOLD=0.25, 
                                  IOU_THRESHOLD=0.45,
                                  pred_gt_matching=pred_gt_matching)
    
    args ={
        'det_dir': 'txt_output/v11m_coco_v2data_PA7_5_classes_500ep_agnostic_conf=0.25',
        'gt_dir': 'v2_malaria_PA9_10_class/test/labels',

        'confusion_matrix': 'confusion_matrix/our_data/' + 'rectangle_' +
            'v11m_coco_v2data_PA7_5_classes_500ep_agnostic_conf=0.25' + '.jpg',

        # 'confusion_matrix': 'confusion_matrix/IML/' + 'rectangle_' +
        #     'v11m_coco_iml_5_classes_500ep_agnostic_conf=0.25' + '.jpg',

        'img_dir': 'v2_malaria_PA9_10_class/test/images',
        'save_dir': 'img_plots/' + 'v11m_coco_v2data_PA7_5_classes_500ep_agnostic_conf=0.25/',
        # 'confusion_matrix': 'draft.png'
    }

    gt_det_img_dict =  load_gt_det_img_folders(args['gt_dir'], args['det_dir'], args['img_dir'])

    for img_name, obj in gt_det_img_dict.items():
        gt, det, img = obj
        cm.process_batch(det, gt, img, img_name, args['save_dir'])
    
    cm.print_matrix()
    cm.plot_matrix(save_path=args['confusion_matrix'], normalize=False, cmap="Reds")