# Summarize
The network generates scores for the presence of each object category in each default box and produces adjustments 
tto the box to better math the object shape.

The core of SSD is predicting category scores and box offsets for a "fixed set of default bounding boxes using
small convolutional filters applied to feature map.

produce predictions of "different scales from feature maps of different scales", and explicitly separate predictions by aspect ratio

## Model
<img width="440" alt="image" src="https://github.com/Developerinsight/Paper_review/assets/123748877/90b1c8e9-b1bd-4a19-92d2-1f1607f812c2">

### Multi-scale feature maps for detection
Add Convolutional feauture layers decreasing in size to detect  at multiple scales.

### Convolutional predictors for detection
using a small convolutional filter to predict object categories.
applying 3x3 convolutional filter per layer.

### SSD Framework
<img width="583" alt="image" src="https://github.com/Developerinsight/Paper_review/assets/123748877/73b754f1-c5ad-44cf-82a9-9319f423133f">

matching two default boxes with the cat and one with the dog, which are treated as postives and the rest as negatives.
yielding (c+4)kmn outputs.(c: object class num, 4: coordinates, k: default box num)

### Loss Function
<img width="474" alt="image" src="https://github.com/Developerinsight/Paper_review/assets/123748877/b6b3b190-2327-4295-8b13-e6d119880e93">

confidence loss + weight x localization loss

<img width="451" alt="image" src="https://github.com/Developerinsight/Paper_review/assets/123748877/d1c88a34-9761-4fcf-aa75-b931b0889fd6">

localization loss is a Smooth L1 loss between the predicted box and the ground truth box parameters.

<img width="473" alt="image" src="https://github.com/Developerinsight/Paper_review/assets/123748877/15b76e4f-2593-4610-bbc3-180609cc863e">

confidence loss is the softmax loss over multiple classes confidences

### Analysis

<img width="511" alt="image" src="https://github.com/Developerinsight/Paper_review/assets/123748877/418dac68-8218-4d54-9b4e-ffbabb91e0b0">

Data augmentation is crucial
More default box shapes is better
Multiple output layers at different resolutions is better

### Else
begining by matching each ground truth box to the default box with the best jaccard overlap higher than a threshold(0.5).

highest negative anchor trains only so that training the background and the ratio between the negatives and positives is at most 3:1.
