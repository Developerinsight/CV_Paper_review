# Very Deep Convolutional Network For Large-Scale Image Recognition
Link: https://arxiv.org/pdf/1409.1556.pdf

## Key Points
### pushing the depth to 16-19 weight layers + small (3x3) convlolution filters(eqaul to receptive field)

#### What is the receptive field?
extract feature pass through each layers.
##### then Why is the smaller 3x3 receptive field effective in vgg?
the reason is 
1. in vgg, it decreases the convolutional layer size as it gets deeper. at start, it detects simple features from layers, and gets abstract, complicated features as it gets deeper.
2. increasing of non-linear. for example, in 10x10 conv layer, 3x3 receptive field activates relu function 25 times, but 7x7 for 9 times. it increases ability to dectect more complicated pattern. 



## Architecture
* Input to first convnet is a fixed-size 224x224 image
* subtract mean rgb value from each pixel to data normalization
* use 3x3 receptive field which is the smallest size to capture the notion of left/down, up/down, center.
* Relu function for conv layers - Relu(x) = max(0, x) => increase non-linear so that we can learn complicated pattern.
  <img width="496" alt="image" src="https://github.com/Developerinsight/Paper_review/assets/123748877/6262676e-6aa2-41c3-b95f-ea884a287d8d">

* Soft max function for fully connected layer => predict probability belonging to each class.

 <img width="296" alt="image" src="https://github.com/Developerinsight/Paper_review/assets/123748877/395159f4-b5ee-453d-b69b-bd6e86ead2b2">


### Training
optimise the multinomial logistic regression objective using mini-batch gradient descent(back propagation)
* multinomial logistic regression: use softmax function to predict probability belonging to classes.
* cross entropy loss for objective function to minimize the loss.
* to minimize, use back propagation, one of the gradient descent.

#### Two approaches
S is the smallest side of rescaled training image.
##### first train the network using s= 256 and train s=384 network with the weights pretrained with s=256
##### <<<
##### rescaled randomly range [Smin, Smax]
=> this confirms that training set augmentation by random scale is helpful.

### Testing
The result is a class score map with the number of
channels equal to the number of classes, and a variable spatial resolution, dependent on the input
image size. 
=> if use big size input image, then we can get bigger class score map compared to small size input image.
Strong point is we can analyze broader, detail region. so if you want to capture small object, then increase the input image size.

### Localisation
average prediction bounding box coordinates, which merges spatially close prediction.
calculate IOU(intersection over union) with ground truth box. 
if it is above 0.5, then bnb prediction is deemed correct.

### Conclusion
it is helpful to increase the improvement if you utilise below.
* Dense evaluation(continuous evaluation of the entire image through all conv layer)
* Multi crop evauation(take the sevral crop image independently, so get a variety of image anaysis)
