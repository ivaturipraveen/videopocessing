import os
import cv2
import torch
import numpy as np
from pathlib import Path
import shutil
import sys
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Add YOLOv5 directory to path (if needed)
sys.path.append('yolov5')

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size  # Note: Check latest utility functions
from utils.augmentations import letterbox

# YOLOv5 predefined classes
YOLOV5_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 
    57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 
    62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

def preprocess_query(query):
    words = word_tokenize(query.lower())
    stop_words = set(stopwords.words('english'))
    interrogative_terms = {'who', 'what', 'where', 'when', 'why', 'how', 'find'}
    meaningful_words = []
    for word, tag in pos_tag(words):
        if word not in stop_words and word not in interrogative_terms and (tag.startswith('N') or tag.startswith('V')):
            meaningful_words.append(word)
    return meaningful_words

def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Extracting frames from {video_path}...")
    progress_step = max(total_frames // 10, 1)
    pbar = tqdm(total=total_frames)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"frame_{count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        count += 1
        pbar.update(1)
        pbar.set_description(f"Extracted: {count}/{total_frames}")

    pbar.close()
    cap.release()
    cv2.destroyAllWindows()
    return count

def detect_objects_yolo(frame_path, model, device, meaningful_words):
    img = cv2.imread(frame_path)
    img0 = img.copy()
    img = letterbox(img, 640, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img, augment=False)
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

    detected_meaningful_frames = []

    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            labels = det[:, -1].cpu().numpy()
            label_names = [model.names[int(label)] for label in labels]

            found_words = set()
            for word in meaningful_words:
                for i, name in enumerate(label_names):
                    if word in name.lower():
                        xyxy = det[i, :4]
                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        cv2.rectangle(img0, c1, c2, (255, 0, 0), 2)
                        found_words.add(word)
            if len(found_words) == len(meaningful_words):
                detected_meaningful_frames.append(frame_path)

    return detected_meaningful_frames, img0

def main(video_path, output_folder, query):
    os.makedirs(output_folder, exist_ok=True)

    frame_count = extract_frames(video_path, output_folder)
    print(f"Total frames extracted: {frame_count}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DetectMultiBackend('yolov5s.pt', device=device)
    model.eval()

    meaningful_words = preprocess_query(query)
    print(f"Meaningful words extracted from query: {meaningful_words}")

    queried_frames_folder = os.path.join(output_folder, "_".join(meaningful_words))
    os.makedirs(queried_frames_folder, exist_ok=True)

    frame_files = sorted([f for f in os.listdir(output_folder) if f.endswith('.jpg')])
    queried_frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(output_folder, frame_file)
        detected_frames, img0 = detect_objects_yolo(frame_path, model, device, meaningful_words)

        if detected_frames:
            queried_frame_path = os.path.join(queried_frames_folder, os.path.basename(frame_path))
            cv2.imwrite(queried_frame_path, img0)
            queried_frames.extend(detected_frames)

    if not queried_frames:
        print(f"No frames containing the queried object(s) '{query}' found.")
        return None

    queried_frames.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))

    first_frame = cv2.imread(queried_frames[0])
    height, width, channels = first_frame.shape

    video_output_path = os.path.join(output_folder, f'output_video_{query}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output_path, fourcc, 25.0, (width, height))

    for frame_path in queried_frames:
        frame = cv2.imread(frame_path)
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()

    print(f"Video created: {video_output_path}")
    return video_output_path

