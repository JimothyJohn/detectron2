#!/usr/bin/python


# Import common libraries
import cv2, mimetypes, argparse
mimetypes.init()

# Import Detectron libraries
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.data import MetadataCatalog
from detectron2.projects import point_rend

# Builds dictionary of class names or keypoints
def BuildDict(cfg):
    # Extract class names from metadata
    classNames = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    # If only one class present then we're doing keypoints
    if len(classNames) == 1:
        classNames = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).keypoint_names
        
    # Add descriptions to class names
    classDict = dict()
    for idx, className in enumerate(classNames):
        classDict[idx]=className

    return classDict
    
# Get input file, output target, and inference type
parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Input file', dest='input', required=True)
parser.add_argument('--stream', help='Mask file', type=bool, dest='stream', default=False)
parser.add_argument('--record', help='Output file', dest='output')
parser.add_argument('--inference', help='Type of inference requested (detection, segmentation, keypoint, panoptic)', dest='inference', default='pointrend')
parser.add_argument('--start', help='Start of video', type=int, dest='start', default=0)
parser.add_argument('--length', help='Length of video', type=int, dest='length', default=256)
args = parser.parse_args()

# Allowable inference types
inferDict = {'detection': 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
            'segmentation': 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
            'panoptic': 'COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml',
            'lvis': 'LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml',
            'keypoint': 'COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml',
            'pointrend': 'COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml',
            'cities': 'Cityscapes/mask_rcnn_R_50_FPN.yaml'}

# Create configuration based on inference type
cfg = get_cfg()
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well 
if args.inference == 'lvis':
    print('Not working yet, sorry! :)')
elif args.inference == 'pointrend':
    point_rend.add_pointrend_config(cfg)
    cfg.merge_from_file("projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
    cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_3c3198.pkl"
else:
    cfg.merge_from_file(model_zoo.get_config_file(inferDict[args.inference]))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(inferDict[args.inference])

# Build class dictionary and load Predictor
classDict = BuildDict(cfg)
predictor = DefaultPredictor(cfg)

# Discern media type
if args.input.startswith('/dev/video'):
    mediaType = "webcam"
else:
    mediaType = mimetypes.guess_type(args.input)[0].split('/')[0]
    print(mediaType)

if mediaType == "image":
    # Read image and save prediction
    image = cv2.imread(f"{args.input}")
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
    outputs = predictor(image)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(args.output, out.get_image()[:, :, ::-1]) # Flip RGB channels

elif mediaType == "video":
    # Start video capture and get attributes
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    success, image = cap.read()
    h, w, c = image.shape
    print(f"Video W/H/FPS: {w}/{h}/{fps}")

    # Start video writer with above attributes
    writer = cv2.VideoWriter(args.output,cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # Start video visualizer
    v = VideoVisualizer(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))

    # Begin processing video
    frame = 0
    while success:
        frame += 1
        success, image = cap.read()
        if frame < args.start*fps:
            continue
        elif frame > (args.start+args.length)*fps: 
            break
        else:
            print(f"Writing frame {frame}")
            outputs = predictor(image)
            out = v.draw_instance_predictions(image, outputs["instances"].to("cpu"))
            writer.write(out.get_image()) 

    cap.release()
    writer.release()

elif mediaType == "webcam":
    # Start video capture and get attributes
    stream = int(args.input[-1])
    cap = cv2.VideoCapture(stream)
    fps = cap.get(cv2.CAP_PROP_FPS)
    success, image = cap.read()
    if not success:
        print('Unable to start stream!')
        exit()

    h, w, c = image.shape
    print(f"Stream W/H/FPS: {w}/{h}/{fps}")

    # Start video visualizer and window
    v = VideoVisualizer(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    cv2.namedWindow('Detectron-LIVE', cv2.WINDOW_AUTOSIZE)

    # Begin processing video
    while success:
        success, image = cap.read()
        outputs = predictor(image)
        out = v.draw_instance_predictions(image, outputs["instances"].to("cpu"))
        cv2.imshow('Detectron-LIVE', out.get_image())
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()

else:
    print('Wrong file type!')
    exit()

'''
instanceBBox = outputs['instances'][0].pred_boxes.tensor.to('cpu').numpy()[0]
instanceName = classDict[outputs['instances'][0].pred_classes.to('cpu').numpy()[0]]
instanceScore = round(outputs['instances'][0].scores.to('cpu').numpy()[0],3)

if args.inference == "panoptic":
    print("'sem_seg' - Total number of blobs (mostly useless)")
    print("'panoptic_seg' - Panoptic seg map + annotation")
    panoArray = outputs['panoptic_seg'][0].to('cpu').numpy()
    print(f"Panoptic array: {panoArray}")
    cv2.imwrite('/scratch/testout.jpg', panoArray*20)

elif args.inference == "segmentation" or args.inference == "pointrend":
    print("'pred_masks' - Boolean segment HxW array for each instance")
    segMask = outputs['instances'][0].pred_masks.to('cpu').numpy()[0]
    print(f"First instance segmentation: {segMask}")

elif args.inference == "keypoint":
    print("'pred_keypoints' - 16-point keypoint output")
    keypointList = outputs['instances'][0].pred_keypoints.to('cpu').numpy()[0]
    for idx, keypoint in enumerate(keypointList):
        print(f"{classDict[idx]} - X: {keypoint[0]}  Y: {keypoint[1]}")
'''