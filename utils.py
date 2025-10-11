import os
import numpy as np
import cv2
import colorsys

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

def load_gt_det_img_folders(gt_folder, det_folder, img_folder):
    """
    Trả về dict {filename: (gt_array, det_array, img_array, W, H)}
    """

    result = {}
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    # duyệt qua tất cả file trong img_folder
    for fname in sorted(os.listdir(img_folder)):
        if not fname.lower().endswith(valid_ext):
            continue  # bỏ qua file không phải ảnh

        img_path = os.path.join(img_folder, fname)
        # thay thế extension ảnh -> .txt cho gt và det
        base_name, _ = os.path.splitext(fname)
        gt_path = os.path.join(gt_folder, base_name + '.txt')
        det_path = os.path.join(det_folder, base_name + '.txt')

        assert os.path.exists(img_path), f"Image file {img_path} does not exist."

        # đọc ảnh và chuyển BGR -> RGB
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        H, W = img.shape[:2]

        # đọc ground-truth
        if os.path.exists(gt_path):
            gt = read_gt(gt_path, W, H)
        else:
            gt = np.empty((0, 6))

        # đọc detection
        if os.path.exists(det_path):
            det = read_det(det_path, W, H)
        else:
            det = np.empty((0, 6))

        result[fname] = (gt, det, img)

    return result

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


def get_class_colors(color_len=20):
    """
    Trả về dict {class_name: (B, G, R)} với màu cách đều nhau.
    """
    colors = {}
    for i in range(color_len):
        # Hue trải đều từ 0 → 1
        h = i / color_len
        s, v = 0.85, 0.65  # độ bão hòa và độ sáng (cố định cho dễ nhìn)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)  # convert HSV -> RGB [0,1]
        colors[str(i)] = (int(b*255), int(g*255), int(r*255))  # OpenCV dùng BGR
    return colors

def plot_boxes_with_labels(boxes, labels, img, save_path, base_scale=1000):
    """
    Vẽ các bounding box và label lên ảnh và lưu.
    Box + chữ sẽ tự scale theo kích thước ảnh.
    """
    im = img.copy()
    H, W = im.shape[:2]

    # Nếu ảnh là RGB thì convert sang BGR cho cv2
    if im.shape[2] == 3 and im[..., 0].mean() >= im[..., 2].mean():
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    # --- scale theo kích thước ảnh ---
    scale = max(H, W) / base_scale
    line_width = max(1, int(2 * scale))          # độ dày khung
    font_scale = 0.5 * scale                     # cỡ chữ
    font_thickness = max(1, int(1 * scale))      # độ dày chữ

    for box, label in zip(boxes, labels):
        cls, x1, y1, x2, y2 = map(int, box[:])

        color = get_class_colors()[str(cls)]

        # Vẽ box
        cv2.rectangle(im, (x1, y1), (x2, y2), color,
                      line_width, cv2.LINE_AA)

        # Tạo background cho label
        w, h = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                               font_scale, font_thickness)[0]
        outside = y1 - h - 3 >= 0
        p1 = (x1, y1 - h - 3 if outside else y1 + 3)
        p2 = (x1 + w + 2, y1 if outside else y1 + h + 3)

        cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # filled background
        cv2.putText(im, label, (p1[0], p2[1] - 2 if not outside else p2[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                    thickness=font_thickness, lineType=cv2.LINE_AA)

    # Lưu ảnh
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, im)
