# gradient-descent-variants-optimization-algorithms
The gradient descent variants optimization algorithms that are widely used by the deep learning community.


Momentum:
--------

SGD has trouble navigating the surface curves much more steeply in one dimension than in another [Sut86], which are common around the local optima.
Momentum [Qia99] is a method that helps accelerate SGD in the relevant direction and dampens oscillations by adding a fraction γ of the update vector of the past time step to the current update vector, as:

θ = θ − νt , with  νt = γ νt−1  +  η ∇θ J(θ)

The momentum parameter γ is usually set to 0.9 or a similar value. As a result, we gain faster convergence and reduced oscillation.


Nesterov accelerated gradient:
Using the momentum method, we push a ball down a hill. The ball accumulates momentum as it rolls downhill, becoming faster and faster on the way. However, a ball rolls down a hill, which blindly following the slope, is unsatisfactory. Nesterov Accelerated Gradient (NAG) [Nes83] is a way to have a smarter ball. So, it knows to slowdown before thehill slopes up again. NAG is defined as:
θ = θ − νt , with  νt = γ νt−1  +  η ∇θ J(θ - γ νt−1)
Similar to the momentum term, NAG parameter γ is set to a value of around 0.9.


Adagrad:
Adagrad [DHS11] is an algorithm for gradient-based optimization that adapts the different learning rate to the parameters, performing larger updates for infrequent parameters and smaller updates for the frequent ones. Therefore, it is well-suited to deal with the sparse data.
As a reminder, the SGD update for every parameter θi at each time step  t  then becomes:  
θt+1,i = θt,i − η ⋅ gt,i   where  gt,i = ∇θ J(θi)
Adagrad method modifies the general learning rate η at each time step  t  for every parameter θi based on the past gradients that have been computed for θi, in its update rule:
θt+1,i = θt,i − [ η / sqrt(Gt,ii+ϵ) ] ⋅ gt,i
Gt ∈ ℝd×d : a diagonal matrix where each diagonal element i,i is the sum of the squares of the gradients w.r.t. θi up to time step t 
and
ϵ : a smoothing term that avoids division by zero (usually on the order of 1e−8)
Since Gt contains the sum of the squares of the past gradients w.r.t. to all parameters θ along its diagonal, one can vectorize the implementation by performing an element-wise matrix-vector multiplication ⊙ between Gt and g:
θt+1 = θt  − [ η / sqrt(Gt+ϵ) ]  ⊙ gt
Notice that, without the square root operation, the algorithm performs much worse.
Adagrad greatly improves he robustness of the SDG approach, but the main weakness of Adagrad algorithm is its accumulation of the squared gradients in the denominator. Since every added term is positive, the accumulated sum keeps growing during the training process.


Adadelta:
Adadelta [Zei12] is an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate. Instead of accumulating all past squared gradients, Adadelta restricts the window of accumulated past gradients to some fixed size w.
The running average of squared gradients:  E[g2]t = γ E[g2]t-1  + (1-γ) g2t
The running average of squared parameter updates:  E[Δθ2]t = γ E[Δθ2]t-1  + (1-γ) Δθ2t
The root mean squared error of the parameter:  RMS[Δθ]t = sqrt(E[Δθ2]t + ϵ)
Since RMS[Δθ]t is unknown, we approximate it with the RMS of parameter updates until the previous step.
Replacing the learning rate η with the RMS[Δθ]t-1  yields the Adadelta update role, as:
θt+1 = θt  +  Δθt   ,with  Δθt = - (RMS[Δθ]t-1) / (RMS[g]t) . gt
Therefore, with Adadelta method, since η has been eliminated from the update role, we do not need to set a default value for the learning rate η.


RMSprop:
Root Mean Squared Prop (RMSprop) (an unpublished, adaptive learning rate method proposed by Geoff Hinton in his Coursera Class) and Adadelta have both developed independently at the same time to resolve Adagrad's radically diminishing learning rates. It divides the learning rate by an exponentially decaying average of squared gradients.
In a nutshell, the RMSprop keeps a moving average of the squared gradient for each parameter, and defined as:
θt+1 = θt  − [ η / sqrt(E[g2]t + ϵ) ]  .  gt
where  E[g2]t = 0.9 E[g2]t-1  + 0.1 g2t
The running average E[g2]t  at time step t  depends  only on the previous average and the current gradient.
Hinton suggests γ to be set to 0.9 and the good default value for the learning rate η  is 0.001.


Adam:
Adaptive Moment Estimation (Adam) [KiB15] is another method that computes adaptive learning rates for each parameter. In addition to storing an exponentially decaying average of past squared gradients (called vt) like Adadelta and RMSprop, Adam also keeps an exponentially decaying average of past gradients (called mt), similar to momentum:
mt = β1 mt-1  +  (1-β1) gt
vt = β2 vt-1 + (1-β2) gt2
mt and vt are estimates of the first moment (the mean) and the second moment (the uncentered variance) of the gradients respectively. As mt and vt  are initialized as vectors of 0's, the authors observed that they are biased towards zero, especially during the initial time steps, and especially when the decay rates are small (i.e. β1 and β2 are close to 1). So, they counteract these biases by computing bias-corrected first and second moment estimates, as:
mˆt = mt / (1-β1t)
vˆt = vt / (1-β2t)
and finally, the Adam update rule is defined as:
θt+1 = θt  − [ η / sqrt(vˆt) + ϵ) ]  .  mˆt 
The good default values; 0.9 for β1, 0.999 for β2, and 10−8 for ϵ.


AdaMax:
The vt factor in the Adam update rule scales the gradient inversely proportionally to the ℓ2 norm of the past gradients (via vt-1) and the current gradient |gt|2:
vt = β2 vt-1 + (1-β2) |gt|2
We can generalize this update to the ℓp norm, as:
vt = β2p vt-1 + (1-β2p) |gt|p
Notice that the norms for large p values generally become numerically unstable, and therefore ℓ1 and ℓ2 norms are most common in practice. However, ℓ∞ also generally exhibits stable behavior. For this reason, the authors propose AdaMax in [KiB15] and show that  vt with ℓ∞ converges to the following more stable value. To avoid confusion with Adam, we use  ut to denote the infinity norm-constrained vt :
ut = β2∞ vt-1 + (1-β2∞) |gt|∞   =  max ( β2 vt-1 , |gt| )
So, the AdaMax update rule is defined as:
θt+1 = θt  −  ( η / ut ) . mˆt 
The good default values are again η=0.002, β1=0.9, and β2=0.999, and we do not need to compute a bias correction for ut (because of its max operation).


Nadam:
Nadam (Nesterov-accelerated Adaptive Moment Estimation) [Doz16] combines Adam and NAG. In order to incorporate NAG into Adam, we need to modify its Momentum term mt.  Adam contributes the exponentially decaying average of past squared gradients vt, while momentum accounts for the exponentially decaying average of past gradients mt.
Let's recall momentum method:
θt+1 = θt − mt , with  mt = γ mt-1  +  η ∇θ J(θ)   and   gt = ∇θt J(θt)
J: the objective function
γ: the momentum decay term
η: the step size
By expanding the above equation we have:
θt+1 = θt − (γ mt−1 + η gt)
So, the momentum involves taking a step in the direction of the previous momentum vector and a step in the direction of the current gradient.
NAG then allows us to perform a more accurate step in the gradient direction by updating the parameters with the momentum step before computing the gradient.
θt+1 = θt − mt , with  mt = γ mt−1  +  η gt     and     gt = ∇θt J(θt - γ mt−1) 
Dozat in [Doz16] proposes to modify NAG the following way: rather than applying the momentum step two times (one time for updating the gradient gt and a second time for updating the parameters θt+1), we apply the look-ahead momentum vector directly to update the current parameters:
θt+1 = θt − (γ mt  +  η gt) , with  mt = γ mt−1  +  η gt     and     gt = ∇θt J(θt) 
Notice that rather than utilizing the previous momentum vector mt−1 as in the equation of the expanded momentum update rule above, we now use the current momentum vector mt to look ahead.  In order to add Nesterov momentum to Adam, we can thus similarly replace the previous momentum vector with the current momentum vector. First, recall that the Adam update rule is the following:
mˆt = mt / (1-β1t)
vˆt = vt / (1-β2t)
and
θt+1 = θt  − [ η / sqrt(vˆt) + ϵ) ]  .  mˆt 
By expanding and replacement of equations, the Nadam update rule is defined as:
θt+1 = θt  − [ η / sqrt(vˆt) + ϵ) ]  .  ( β1mˆt   +  ( (1-β1) gt / (1-β1t) ) )



References:
[Sut86] Sutton, R. S. (1986). Two problems with backpropagation and other steepest-descent learning procedures for networks. Proc. 8th Annual Conf. Cognitive Science Society.

[Qia99] Qian, N. (1999). On the momentum term in gradient descent learning algorithms. Neural Networks : The Official Journal of the International Neural Network Society, 12(1), 145–151.

[Nes83] Nesterov, Y. (1983). A method for unconstrained convex minimization problem with the rate of convergence o(1/k2). Doklady ANSSSR (translated as Soviet.Math.Docl.), vol. 269, pp. 543– 547

[DHS11] Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. Journal of Machine Learning Research, 12, 2121–2159.

[Zei12] Zeiler, M. D. (2012). ADADELTA: An Adaptive Learning Rate Method.

[KiB15] Kingma, D. P., & Ba, J. L. (2015). Adam: a Method for Stochastic Optimization. International Conference on Learning Representations, 1–13.

[Doz16] Dozat, T. (2016). Incorporating Nesterov Momentum into Adam. ICLR Workshop, (1), 2013–2016.
