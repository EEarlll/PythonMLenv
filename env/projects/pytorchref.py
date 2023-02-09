"""

@ artificial intelligence
- patterns between input and outputs

used when:
1. problems with long rules
2. changing environments
3. large collections of data

not used when;
1. explanation
2. output is clearly defined
3. small collections of data  

# machine learning
- Structured data ex. excels etc.


algorithms
1. Random forest
2. Gradient boosted models


# deep-learning
- Unstructured data ex. audio, images


algorithms
1. Neural network
- input converted to tensor put to neural network to output
2. Convolutional neural network(CNN)
- used in recognizing patterns in images 

# TYPES:
1. Classification 
    1.1 Binary classification - true or false 1/2, catvsdog , spam/notspam
    1.2 Multiclass Classification - 1 label to each ex. steak, fish, etc
    1.3 Multilabel Classification - multiple label to each  ex. machine learning, artificial intelligence, etc
2. Computer Vision
    2.1 Binary Classification
    2.2 Multiclass Classification
    2.3 Object Detection
    2.4 Segmentation


@ types of learning
supervised - data w/ labels
unsupervised - data w/o labels
transfer learning - model with already learned data
reinforcement learning - train model with rewards

# dtypes
- represents how much details a memory a take
lower precision dtypes takes less memory and faster calculation
more precision dtypes takes more memory and slower calculation

# manipulation tensors: 

matrix multiplication: 
1. inner dimension must be the same : (x, inner) @ (inner, y)
    (3,2) @ (3,2) not work
    (2,3) @ (3,2) work
    (3,2) @ (2,3) work
2. resulting has the shape of outer dimension : (outer, x) @ (y, outer)
    (2,3) @ (3,2) work
3. Ex.
    img_batch size : ([32, 3, 64, 64]) 
    hidden : 10 
    matrix multiplication bc last layer linear
    mat1 and mat2 shapes cannot be multiplied (32x2560 and 10x3) -> (32x2560 and 2560x3)

# improving model
1. Add more layer
    1.1 nn.linear
    1.2 nn.Conv2d
    1.3 nn.MaxPool2d
2. Add more hidden units
    2.1 in & out features if linear
3. Fit for longer
    3.1 epoch
4. Change activation functions
    4.1 Non-linear Activation function:
            ReLU - takes an input if negative = 0 if positive = same
5. Change learning rate
    5.1 lr 
6. Change loss functions
    6.1 CrossEntropy, BCEWithLogitsLoss, L1Loss
7. Change optimizer
    7.1 SGD
    7.2 Adam
8. Change parameters


# Evaluation methods
1. Accuracy
2. Precision 
3. Recall
4. F1-score
5. Confusion matrix
6. Classification report

# Pytorch Domain Libraries
1. TorchVision - Images
2. TorchText - Text
3. TorchAudio - Audio
4. TorchRec - Recommendations
etc. refer to documentation

# Dealing with Overfitting & Underfitting
1. Overfitting:
    1.0 <- DOES NOT ALWAYS WORK EXPERIMENT ->
    1.1 Get more data
    1.2 Data augmentation
    1.3 Get better data
    1.4 Use Transfer learning
    1.5 Simplify model ex. remove extra layers, hidden units
    1.6 Use learning rate decay - slowly decrease lr overtime 
    1.7 Use early stopping - stop before it overfit
2. Underfitting:
    2.0 <- DOES NOT ALWAYS WORK EXPERIMENT ->
    2.1 Add more layer/units
    2.2 Tweak learning rate, maybe too high 
    2.3 train for longer epochs
    2.4 Use transfer learning - using other model prelearned patterns
    2.5 Use less regularization  - preventing overfitting too much


# Pretrained model
1. Get a pretrained model
    1.1 higher parameter means more performance in tradeoff to speed
2. Setup pretrained get weights 
3. Freeze layer
4. Replace last layer to output num_classes
5. Retrain

# Object Detection
1. get a pretrained model
2 setup dataset 
    2.1 create dataset class
    2.2 get img path list and label path list
    2.3 __getitem__ - convert each imagepath[idx] to tensor
    2.4 get bounding box coordinates xmin ymin xmax ymax
    2.5 return img(in tensor) , target(bounding boxes & other info dict)
    2.7 __len__ return len(img path list)
3. dataloader use collate_fn if __getitem__ returns list or tuple
4. finetune pretrained model
5. train prediction gives loss value 
6. validate using cv2/PIL
    6.1 get img path and open
    6.2 convert img to tensor 
        6.2.1 change to rgb format
        6.2.2 change to (N,C,H,W) Format
    6.3 get boxes and score 
    6.4 create rectangle on image based on conditional score


"""