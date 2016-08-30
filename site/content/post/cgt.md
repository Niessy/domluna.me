+++
date = "2015-09-08T12:47:23-04:00"
draft = true
title = "Neural Networks with CGT"

+++

[CGT](), also known as Computational Graph Toolkit is a new Python library which
automatically does reserve-mode differentiation on a graph. If you're not
familiar with reverse-mode differentiation (quite the mouthful) and I'd
highly suggest reading Christopher Colah's [post on it]().

### Alternatives

So before we get into a little bit of CGT, it's a good idea to reflect on
why this might be useful in the first place.

[Theano]() is the closest comparison one can make to CGT, in fact, CGT is
meant to be a clear upgrade from Theano. The most pressing issue of Theano
is the compilation time, that is the time taken for Theano to create the
graph and perform reverse-mode differentiation. It's not uncommon if your
model is big/complex enough that this takes several minutes. Seeing as how
models trending towards becoming more complex, this does not bode well
for users.

![Compilation Pic]()

Torch and Caffe are other well known alternatives. The issue here is they do
not perform automatic reverse-mode differentiation. If you create a new module/computation
you would need to implement this yourself. 

Not fun! 

It's also worth mentioning that Torch and Caffe are deep learning specific. While it's no 
question the inspiration behind CGT is for deep learning models, CGT is general with respect
to computations on graphs.

Ok, with the preamble out of the way let's dive into CGT!

### CGT

What I want to do here is introduce the basic concepts in CGT by addressign the following questions.

1. How do I update weights/parameters of my model?
2. How do I feed data into the model?
3. How do I train the model?

TODO:

Link to notebook?
