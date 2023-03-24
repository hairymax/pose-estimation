import argparse
import cv2
import torch
import numpy as np
from torchvision import transforms
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, xywh2xyxy
from utils.plots import output_to_keypoint, plot_one_box, plot_skeleton_kpts

# Path to the weights file
POSE_WEIGHTS = 'weights/yolov7-w6-pose.pt'

# Select the device to run the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Selected Device : ', device)

# Load the pose estimation model
model = attempt_load(POSE_WEIGHTS, map_location=device)  # load model
model.eval()


def make_pose_prediction(model, img:cv2.Mat) -> list:
    ''' Make prediction with pretrained yolo pose estimator `model` on image `img`
    '''
    # Resize and pad image while meeting stride-multiple constraints
    img_ = letterbox(img, 960, stride=64, auto=True)[0]
    resized_shape = img_.shape[0:2]
    # Transform image to model readable structure
    img_ = transforms.ToTensor()(img_)
    img_ = torch.tensor(np.array([img_.numpy()]))
    img_ = img_.to(device).float()
    with torch.no_grad():
        output, _ = model(img_)
    # Filter predictions
    output = non_max_suppression_kpt(output, 0.25, 0.65, 
                                     nc=model.yaml['nc'], 
                                     nkpt=model.yaml['nkpt'], 
                                     kpt_label=True)
    output = output_to_keypoint(output)
    # scale to original image shape
    output = scale_pose_output(output, resized_shape, img.shape[0:2])
    return output


def scale_pose_output(output, resized_shape:tuple, original_shape:tuple, is_padded:bool=True):
    ''' Scale yolo pose estimator output coordinates of bbox and keypoints
    from `resized_shape` to `original_shape` 
    '''
    scaled_output = output.copy()
    scale_ratio = resized_shape[1] / original_shape[1], resized_shape[0] / original_shape[0]      
    if is_padded:
        # remove padding
        pad_scale = min(scale_ratio)
        padding = ( resized_shape[1] - original_shape[1] * pad_scale ) / 2, (
                    resized_shape[0] - original_shape[0] * pad_scale ) / 2
        scale_ratio = (pad_scale, pad_scale)
        
        scaled_output[:, 2] -= padding[0]     # x_c unpadding
        scaled_output[:, 3] -= padding[1]     # y_c unpadding
        scaled_output[:, 7::3] -= padding[0]  # x_kpts unpadding
        scaled_output[:, 8::3] -= padding[1]  # y_kpts unpadding
    
    scaled_output[:, [2, 4]] /= scale_ratio[0]
    scaled_output[:, [3, 5]] /= scale_ratio[1]
    scaled_output[:, 7::3] /= scale_ratio[0]
    scaled_output[:, 8::3] /= scale_ratio[1]

    return scaled_output

def plot_pose_prediction(img : cv2.Mat, pred : list, thickness=2, 
                         show_bbox:bool=True) -> cv2.Mat:
    """draw predicted pose"""
    bbox = xywh2xyxy(pred[:,2:6])
    for idx in range(pred.shape[0]):
        plot_skeleton_kpts(img, pred[idx, 7:].T, 3)
        if show_bbox:
            plot_one_box(bbox[idx], img, line_thickness=thickness)


# Function for processing a video file frame by frame
def process_video(input_path, output_path):
    # Open the input video file and extract its properties
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # Create VideoWriter object 
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    #  Processing a video file frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            pred = make_pose_prediction(model, frame)
            plot_pose_prediction(frame, pred, show_bbox=False)
            out.write(frame)
            cv2.imshow('Pose estimation', frame)
        else:
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    # Release VideoCapture and VideoWriter
    cap.release()
    out.release()
    # Close all frames and video windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to input video file')
    parser.add_argument('-o', '--out', type=str, required=True, help='Path to output video file')
    args = parser.parse_args()
    process_video(args.input, args.out)