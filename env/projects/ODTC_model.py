import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import cv2

class ODTC_model:
    """ODTC_model pretrained using fasterrcnn_resnet50_fpn model"""
    def __init__(self,device = 'cuda') -> None:
        self.device = device

        # build model
        self.model_6 = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = self.model_6.roi_heads.box_predictor.cls_score.in_features
        num_classes = 2
        self.model_6.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.model_6.to(device)

        # load state_dict 
        path_rel = r'C:\Users\earle\PythonMLenv\env\projects\Models\03_ODTC.pt'
        self.model_6.load_state_dict(torch.load(path_rel))
        self.model_6.eval()

    def pred(self, img_path):
        image = cv2.imread(img_path)
        orig_image = image.copy()
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        image = np.transpose(image, (2,0,1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float).to(self.device)
        image = torch.unsqueeze(image, 0)

        with torch.inference_mode():
            self.model_6.eval()
            output = self.model_6(image.to(self.device))

        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in output]
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()

        return outputs, boxes , scores , orig_image
    
    def show_predbox(self, output, orig_image , pref_score = 0.8 ):
            boxes = output[0]['boxes'].data.numpy()
            scores = output[0]['scores'].data.numpy()

            boxes = boxes[scores >= pref_score].astype(np.int32)
            draw_boxes = boxes.copy()
            pred_classes = [i for i in output[0]['labels'].cpu().numpy()]

            for k, box in enumerate(draw_boxes):
                cv2.rectangle(orig_image,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            (0,0,255), 2)
                cv2.putText(orig_image, str(pred_classes[k]),
                            (int(box[0]), int(box[1]-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),
                            2, lineType=cv2.LINE_AA)

                cv2.imshow('Prediction', orig_image)
                cv2.waitKey(10000)
                cv2.destroyAllWindows()
    
    def predbbox(self, img_path, pref_score = 0.8):
        image = cv2.imread(img_path)
        orig_image = image.copy()
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        image = np.transpose(image, (2,0,1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float).to(self.device)
        image = torch.unsqueeze(image, 0)

        with torch.inference_mode():
            self.model_6.eval()
            output = self.model_6(image.to(self.device))

        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in output]
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()

        boxes = boxes[scores >= pref_score].astype(np.int32)
        draw_boxes = boxes.copy()
        pred_classes = [i for i in output[0]['labels'].cpu().numpy()]
        for k, box in enumerate(draw_boxes):
            cv2.rectangle(orig_image,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        (0,0,255), 2)
            cv2.putText(orig_image, str(pred_classes[k]),
                        (int(box[0]), int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),
                        2, lineType=cv2.LINE_AA)
            cv2.imshow('Prediction', orig_image)
            cv2.waitKey(10000)
            cv2.destroyAllWindows()

