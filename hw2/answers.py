r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
1. For a single sample the Jacobian matrix is of size (2048, 1024) - (output dim, input dim). When X is a tensor with batch size 128 
this adds 2 dimension to the Jacobian which would bring the size to (128, 2048, 128, 1024).
<br><br>
2. The number of parameters in the matrix is $ 128 * 2048 * 128 * 1024 = 68719476736
$. <br> Each parameter takes up 32 bits
so in total we would be using $ 32 * 68719476736 $ bits. <br>
In gigabytes that is: $ \frac{(32 * 68719476736)}{1024^3} = 2048 $ gigabytes


"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 0.1, 0.02, 0.01

    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg = 0.1, 0.1, 0.01, 0.00005, 0.001

    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # ====== YOUR CODE: ======
    wstd, lr = 0.1, 0.01
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
1. The graphs match what we expected to see which was overfitting in the case of the no-dropout model. Since we are working
on a small dataset with a large number parameters we are likely to run into overfitting. 
The dropout models performed worse on the train set compared to the no-dropout, this is due to the overfitting. 
But we see that on the test set the dropout models are performing much better, which is precicely the definition of overfitting. 

2. The 0.4 dropout model performed the best in test accuracy, and second best in train accuracy. This seems to be the prefered hyperparameter
for the dropout. It could be that using 0.8 as dropout parameter is too small in our case.
But as we continue with more epochs, the 0.8 dropout model gets closet to test accuracy to the 0.4 model and eventually beats out the 0.4 model.
So we see there is a trade off between weights being learned and epochs the model needs for trianing. With less weights (0.8 model) we need longer epochs.
With more weights (0.4) we can get good results in half the epoch time.
"""

part2_q2 = r"""
Yes using cross-entropy loss function it is possible for test and train loss to increase simultaneously.
For classification model that is predicting the best class, we define best class to be the maximum values of all the predicted vector, which is the size of the number of classes.
However, the cross-entropy loss is looking at the all the values in the predicted vector. The max accuracy index (based on max value) may not be the same as the min loss index (based on all values).
In this situation, we may see a case where both losses are increasing for a couple epochs.
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
1. Number of parameters:
- For regular block: $256⋅256⋅3⋅3+256⋅256⋅3⋅3=1,179,648$<br>
- For bottleneck block: $256⋅64⋅1⋅1+64⋅64⋅3⋅3+64⋅256⋅1⋅1=69,632$<br>

2. Number of floating point operations: number of parameters * width * length
- For regular block:$1179648$ * width * length<br>
- For bottleneck block: $69632$ * width * length<br>


3. The bottleneck block is able to combine input spatially and across feature maps.  While the regular block has a higher ability to combine input spatially (within feature maps).
"""


part3_q2 = r"""
In general, deeper networks should yield better accuracy up until a point where we have too deep a network and there is the vanishing gradient. 
In the vanishing gradient problem because we have too many layers to backpropogate the change to the gradient as we move backward before negligible and the network is not able to train the longer distance weights.
In our experiment, we can see that with L = 4 (medium depth) performs best on test and train accuracy. 
We think this is because with L= 8 or 16 we run into vanishing gradient. As we can se in  the graphs
L = 8 and 16 werent even trainable. To resolve this, we suggest to change the netowrk architecture and run with a resudial network with skip connections or perform batch normalization.
"""

part3_q3 = r"""
In this experiment, we tested on a variery of channel sizes where as in 1.1 we used only K = 32/64. Here we see that
a larger K performed better on a network with the same depth. 
We see again that a deeper CNN network performs better; the L = 4 networks had higher accuracy than the L = 2 networks.
Once we reach L = 8 none of the models are properly trained. The accuracy per epoch ranges from
9 to 10.5% and varies greatly between each epoch. This may be a result from the vanishing gradient problem where the weights can not be properly learned.
"""

part3_q4 = r"""
Here too we see the effect of the vanishing gradient problem for L = 3 and 4. In these cases the network is depth 9 or 12 which is already too deep for the model to learn properly
The best model was with a 3 cnn layer network with channel sizes [64, 128, 256]. For shorter networks like this it seems 
like larger channel sizes will yield better results, compared to experiemnt 1.1 which used only 32 or 4 chanel sizes only.
"""

part3_q5 = r"""
In the ResNet architecture we are able to train on deeper networks. This is because the skip connections help us overcome the vanishing gradient problem. 
In experiment 1.1 we saw that L = 8 and L = 16 were not trainable. Here even deeper networks are trained and get to higher accuracy with maximum test accuracy 
reaching 73%. 
In the ResNet architecture with varryings channels size, we are able to train on a deeper network compared to CNN network in 1.3.
"""

part3_q6 = r"""
In our modified network we decided to create a ResNet network with batch normalization and dropout to see if we can increase accuracy move and improve runtime. We used dropout = 0.5
After 10 epochs we were able to reach test accuracy of 70%. This was achieved with L = 3 model with channel sizes of 32, 64, 128. 
We did not encounter the issue of vanishing gradient because we are using the skip connections. 
However, we were suprised to see that the shortest depth model L = 3 performed the best here. L = 6 and 9 did not achieve significant accuracy results.
Compared to the CNN models in the previous section we were able to successfully train on deeper network. But maybe in this case, we do not need such deep network to achieve good results.
"""
# ==============
