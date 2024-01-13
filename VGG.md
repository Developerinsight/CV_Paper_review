# Very Deep Convolutional Network For Large-Scale Image Recognition

## Key Points
### pushing the depth to 16-19 weight layers + small (3x3) convlolution filters(eqaul to receptive field)

#### What is the receptive field?
extract feature pass through each layers.
##### then Why is the smaller 3x3 receptive field effective in vgg?
the reason is 
1. in vgg, it decreases the convolutional layer size as it gets deeper. at start, it detects simple features from layers, and gets abstract, complicated features as it gets deeper.
2. increasing of non-linear. for example, in 10x10 conv layer, 3x3 receptive field activates relu function 25 times, but 7x7 for 9 times. it increases ability to dectect more complicated pattern. 



## Architecture
Input to first convnet is a fixed-size 224x224 image
subtract mean rgb value from each pixel to data normalization
use 3x3 receptive field which is the smallest size to capture the notion of left/down, up/down, center.
Relu function for conv layers - Relu(x) = max(0, x) => increase non-linear so that we can learn complicated pattern.
Soft max function for fully connected layer => predict probability belonging to each class.


### Training
optimise the multinomial logistic regression objective using mini-batch gradient descent(back propagation)
* multinomial logistic regression: use softmax function to predict probability belonging to classes.
* cross entropy loss for objective function to minimize the loss.
* to minimize, use back propagation, one of the gradient descent.
