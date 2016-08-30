+++
date = "2015-09-30T15:03:58-04:00"
draft = true
comments = true
title = "Implementing a Neural Network"

+++

### Feedforward Networks

The simplest form a neural network is a feedforward network. The following is a single layer of a feedforward network. It consists
of four parts as shown in the following function.

\[
f(\sigma, x, W, b) = \sigma(xW + b)
\]

- $x$ is a vector of size (1, input size)

- $W$ is a matirx of size (input size, output size), the output size is the
input size of the next layer.

- $b$ is a vector of size (1, output size)

Let's decode this equation:

1. $\sigma$ is the activiation function. Typically a [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function), [tanh](http://functions.wolfram.com/ElementaryFunctions/Tanh/introductions/Tanh/ShowAll.html) or [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
function. It's applied elementwise.

2. $x$ is the input (your data) being feed into the network. While literature typically represents $x$ as a vector, in practice it's
a matrix. If we send our input through in batches we train our model
faster.

3. $W$ and $b$ are weights/parameters, these are the values being
optimized to our objective when we call a [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) routine.

If we link 2-3 of these layers together we can already do some
pretty neat stuff, like classify [handwritten digits](http://yann.lecun.com/exdb/mnist/).

![MNIST](/images/mnist.jpg)
