import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def box_iou_calc(boxes1, boxes2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Array[N, 4])
        boxes2 (Array[M, 4])
    Returns:
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2

    This implementation is taken from the above link and changed so that it only uses numpy..
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = np.prod(np.clip(rb - lt, a_min=0, a_max=None), 2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


class DetectionConfusionMatrix:
    def __init__(self, num_classes: int, class_name: list[str], CONF_THRESHOLD=0.3, IOU_THRESHOLD=0.5):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
        self.num_classes = num_classes
        self.class_name = class_name
        if len(class_name) != num_classes + 1:
            raise ValueError("Number of class names must be equal to num_classes + 1")
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.IOU_THRESHOLD = IOU_THRESHOLD

    def process_batch(self, detections, labels: np.ndarray):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        gt_classes = labels[:, 0].astype(np.int16)

        try:
            detections = detections[detections[:, 4] > self.CONF_THRESHOLD]
        except (IndexError, TypeError):
            for i, label in enumerate(labels):
                gt_class = gt_classes[i]
                self.matrix[self.num_classes, gt_class] += 1  # FP (not detected)
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

        for i, label in enumerate(labels):
            gt_class = gt_classes[i]
            match = all_matches[all_matches[:, 0] == i]
            if match.shape[0] == 1:
                det_idx = int(match[0, 1])
                pred_class = detection_classes[det_idx]
                self.matrix[pred_class, gt_class] += 1  # TP
            else:
                self.matrix[self.num_classes, gt_class] += 1  # FN

        for i, detection in enumerate(detections):
            match = all_matches[all_matches[:, 1] == i]
            if match.shape[0] == 0:
                pred_class = detection_classes[i]
                self.matrix[pred_class, self.num_classes] += 1  # FP

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

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, label_nn = self.num_classes, len(self.class_name)-1  # number of classes, names
        seaborn.set_theme(font_scale=6.5 if nc < 50 else 0.8)  # for label size
        labels = (0 < label_nn < 99) and (label_nn == nc)  # apply names to ticklabels
        xticklabels = list(self.class_name) if labels else "auto"
        yticklabels = list(self.class_name) if labels else "auto"
        
       
        seaborn.heatmap(
                array,
                ax=ax,
                annot=nc < 30,
                annot_kws={"size": 20},
                cmap=cmap,
                fmt=".2f" if normalize else ".0f",
                square=True,
                vmin=0.0,
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                cbar=False,
                norm=LogNorm(),
            ).set_facecolor((1, 1, 1))
        title = "Confusion Matrix" + " Normalized" * normalize
        ax.set_xlabel("True", fontsize=18)
        ax.set_ylabel("Predicted", fontsize=18)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
        # ax.set_title(title)
        plot_fname = save_path
        if not os.path.exists(os.path.dirname(plot_fname)):
            os.makedirs(os.path.dirname(plot_fname))
        fig.savefig(plot_fname, dpi=500, bbox_inches='tight')
        plt.close(fig)

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    """
    Convert normalized bounding box coordinates to pixel coordinates.

    Args:
        x (np.ndarray | torch.Tensor): The bounding box coordinates.
        w (int): Width of the image.
        h (int): Height of the image.
        padw (int): Padding width.
        padh (int): Padding height.

    Returns:
        y (np.ndarray | torch.Tensor): The coordinates of the bounding box in the format [x1, y1, x2, y2] where
            x1,y1 is the top-left corner, x2,y2 is the bottom-right corner of the bounding box.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = np.empty_like(x, dtype=np.float32)  # faster than clone/copy
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y

def read_gt(path, W, H):
    """
    Đọc ground truth file và chuyển về ndarray (N, 5): [class, x1, y1, x2, y2]
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.zeros((0, 5))

    data = np.loadtxt(path).reshape(-1, 5)
    boxes = xywhn2xyxy(data[:, 1:5], W, H)
    classes = data[:, 0:1]
    return np.concatenate([classes, boxes], axis=1)

def read_det(path, W, H):
    """
    Đọc detection file và chuyển về ndarray (N, 6): [x1, y1, x2, y2, conf, class]
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.zeros((0, 6))

    data = np.loadtxt(path).reshape(-1, 6)
    boxes = xywhn2xyxy(data[:, 1:5], W, H)
    # boxes = data[:, 1:5]
    confs = data[:, 5:6]
    classes = data[:, 0:1]
    return np.concatenate([boxes, confs, classes], axis=1)

def load_gt_and_det_folders(gt_folder, det_folder, W, H):
    """
    Trả về dict {filename: (gt_array, det_array)}
    Nếu file trong gt_folder không có file tương ứng trong det_folder,
    thì det_array sẽ là mảng rỗng (0,6)
    """
    result = {}

    for fname in sorted(f for f in os.listdir(gt_folder) if f.endswith('.txt')):
        gt_path = os.path.join(gt_folder, fname)
        det_path = os.path.join(det_folder, fname)

        gt = read_gt(gt_path, W, H)

        if os.path.exists(det_path):
            det = read_det(det_path, W, H)
        else:
            det = np.empty((0, 6))

        result[fname] = (gt, det)

    return result

if __name__ == "__main__":
    W, H = 3072, 2048 # for our 4k images
    # W, H = 1280, 960 # for our IML images
    # W, H = 1944, 1383 # for our BBBC041 images
    class_name = ["R", "T", "S", "G", "Un", "Leu", "background"]
    cm = DetectionConfusionMatrix(num_classes=len(class_name)-1, 
                                  class_name=class_name, 
                                  CONF_THRESHOLD=0.25, 
                                  IOU_THRESHOLD=0.45)
    
    # args ={
    #     'det_dir': 'txt_output/v11s_coco_v2data_PA7_5_classes_500ep_agnostic_conf=0.25',
    #     'gt_dir': 'v2_malaria_PA7_5_class/test/labels',
    #     'confusion_matrix': 'v11s_coco_v2data_PA7_5_classes_500ep_agnostic_conf=0.25' + '.jpg'
    #     # 'confusion_matrix': 'draft.png'
    # }

    args ={
        # 'det_dir': 'txt_output/v11m_coco_v2data_PA7_5_classes_500ep_agnostic_conf=0.25',
        'det_dir': 'txt_output/trong_v11m_v2data_5_classes',
        'gt_dir': 'v2_malaria_PA7_5_class/test/labels',
        # 'confusion_matrix': 'confusion_matrix/our_data/v11m_coco_v2data_PA7_5_classes_500ep_agnostic_conf=0.25' + '.jpg'
        'confusion_matrix': 'confusion_matrix/our_data/trong_v11m_v2data_5_classes' + '.jpg'
        # 'confusion_matrix': 'draft.png'
    }

    gt_det_dict =  load_gt_and_det_folders(args['gt_dir'], args['det_dir'], W, H)
    
    for image, boxes in gt_det_dict.items():
        gt, det = boxes
        cm.process_batch(det, gt)
    
    cm.print_matrix()
    cm.plot_matrix(save_path=args['confusion_matrix'], normalize=False, cmap="Reds")
    cm.compute_PR_from_matrix(cm.matrix)