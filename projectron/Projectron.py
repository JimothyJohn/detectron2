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

# Returns webcam, video, or image
def MediaCheck(directory):
    if directory.startswith('/dev/video'):
        return "webcam"
    else:
        return mimetypes.guess_type(directory)[0].split('/')[0]
    
# Allowable inference types
inferDict = {'detection': 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
            'segmentation': 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
            'panoptic': 'COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml',
            'lvis': 'LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml',
            'keypoint': 'COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml',
            'pointrend': 'projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml',
            'cities': 'Cityscapes/mask_rcnn_R_50_FPN.yaml'}

class Projectron:
    def __init__(self, inference):
        if inference not in inferDict.keys():
            print('Choose detection, segmentation, panoptic, keypoint, pointrend, or cities')
        
        self.inference = inference
        # Create configuration based on inference type
        self.cfg = get_cfg()
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well 
        if inference == 'lvis':
            print('Not working yet, sorry! :)')
        elif inference == 'pointrend':
            point_rend.add_pointrend_config(self.cfg)
            self.cfg.merge_from_file(inferDict[inference])
            self.cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_3c3198.pkl"
        else:
            self.cfg.merge_from_file(model_zoo.get_config_file(inferDict[inference]))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(inferDict[inference])

        # Build class dictionary and load Predictor
        self.classDict = BuildDict(self.cfg)
        self.predictor = DefaultPredictor(self.cfg)

    def RecordVideo(self, input, output, start=0, length=256):
        mediaType = MediaCheck(input)
        if mediaType == "video":
            # Start video capture and get attributes
            cap = cv2.VideoCapture(input)
        elif mediaType == "webcam":
            stream = int(input[-1])
            cap = cv2.VideoCapture(stream)
        else:
            print('Wrong media type!')
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        success, image = cap.read()
        h, w, _ = image.shape
        print(f"Video W/H/FPS: {w}/{h}/{fps}")

        # Start video writer with above attributes
        writer = cv2.VideoWriter(output,cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        # Start video visualizer
        v = VideoVisualizer(MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))

        # Begin processing video
        frame = 0
        while success:
            frame += 1
            success, image = cap.read()
            if frame < start*fps:
                continue
            elif frame > (start+length)*fps: 
                break
            else:
                print(f"Writing frame {frame}")
                outputs = self.predictor(image)
                out = v.draw_instance_predictions(image, outputs["instances"].to("cpu"))
                writer.write(out.get_image()) 

        cap.release()
        writer.release()
        print(f"Wrote {output}!")
        return

    def RecordImage(self, input, output):
        if MediaCheck(input) != "image":
            print('Choose an IMAGE file...')
            return
        
        # Read image and save prediction
        image = cv2.imread(f"{input}")
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.0)
        outputs = self.predictor(image)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(output, out.get_image()[:, :, ::-1]) # Flip RGB channels
        
    def StreamVideo(self, input):
        mediaType = MediaCheck(input)
        if mediaType == "video":
            # Start video capture and get attributes
            cap = cv2.VideoCapture(input)
        elif mediaType == "webcam":
            stream = int(input[-1])
            cap = cv2.VideoCapture(stream)
        else:
            print('Wrong input type!')
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        success, image = cap.read()
        if not success:
            print('Unable to start stream!')
            exit()

        h, w, c = image.shape
        print(f"Stream W/H/FPS: {w}/{h}/{fps}")

        # Start video visualizer and window
        v = VideoVisualizer(MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
        cv2.namedWindow('Detectron-LIVE', cv2.WINDOW_AUTOSIZE)

        # Begin processing video
        while success:
            success, image = cap.read()
            self.outputs = self.predictor(image)
            out = v.draw_instance_predictions(image, outputs["instances"].to("cpu"))
            cv2.imshow('Detectron-LIVE', out.get_image())
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        return

    def StreamData(self, instanceID=0, datapoint="class"):
        mediaType = MediaCheck(input)
        if mediaType == "video":
            # Start video capture and get attributes
            cap = cv2.VideoCapture(input)
        elif mediaType == "webcam":
            stream = int(input[-1])
            cap = cv2.VideoCapture(stream)
        else:
            print('Wrong input type!')
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        success, image = cap.read()
        if not success:
            print('Unable to start stream!')
            exit()

        h, w, _ = image.shape
        print(f"Stream W/H/FPS: {w}/{h}/{fps}")

        # Start video visualizer and window
        v = VideoVisualizer(MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
        cv2.namedWindow('Detectron-LIVE', cv2.WINDOW_AUTOSIZE)

        # Begin processing video
        while success:
            success, image = cap.read()
            outputs = self.predictor(image)
            if datapoint == "class":
                className = self.classDict[outputs['instances'][instanceID].pred_classes.to('cpu').numpy()[0]]
                print(f"Class (Index {instanceID}):{className}")
                return className
            elif datapoint == "bbox":
                instanceBBox = outputs['instances'][0].pred_boxes.tensor.to('cpu').numpy()[0]
                print(f"Location (Index {instanceID}):{instanceBBox}")
                return instanceBBox

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        return 
'''

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