import torchvision
import torch

class GLP_model:
    """GLP_model pretrained using vgg19_bn model"""
    def __init__(self, device = 'cuda') -> None:
        # model
        self.device = device
        weights = torchvision.models.VGG19_BN_Weights.DEFAULT
        self.auto_transform = weights.transforms()
        self.model_5 = torchvision.models.vgg19_bn(weights = weights).to(device)
        self.class_Names = ['animal giraffe', 'animal lion', 'animal penguin']
        # freeze layer
        for i in self.model_5.features.parameters():
            i.requires_grad = False

        self.model_5.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5, inplace=False),
            torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5, inplace=False),
            torch.nn.Linear(in_features=4096, out_features=3, bias=True)
        ).to(device)
        
        # load state_dict
        path_rel = r'C:\Users\earle\PythonMLenv\env\projects\Models\02_Giraffe_Lion_Penguin.pt'
        self.model_5.load_state_dict(torch.load(path_rel))

    def pred(self, img_path):
    # predictions 
        if img_path:
            target_image = torchvision.io.read_image(str(img_path)).type(torch.float32) / 255
            # add a batch size (NCHW)
            target_image = target_image.unsqueeze(0).to(self.device)
            if target_image.shape[1] != 3:
                target_image = target_image[:,:3,:,:]

            if self.auto_transform:
                target_image = self.auto_transform(target_image)
            self.model_5.to(self.device)
            self.model_5.eval()
            with torch.inference_mode():
                target_image_logits = self.model_5(target_image)
                target_image_probs = torch.softmax(target_image_logits.squeeze(), dim =0)
                target_image_pred = torch.argmax(target_image_probs, dim =0).cpu()    

            return self.class_Names[target_image_pred]