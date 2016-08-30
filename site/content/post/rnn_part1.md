+++
date = "2015-10-08"
title = "RNNs Part 1: Creating a Cell"
tags = ["RNN", "Deep Learning", "NLP"]
math = true
+++

> Assumes knowledge of basic feedforward networks.

This is the first post in an n-part series about [Recurrent Neural
Networks](https://en.wikipedia.org/wiki/Recurrent_neural_network) (RNNs), the n is TBD.

## Why RNNs?

RNNs are useful when our inputs and/or outputs are of variable length. This has made them
particularly good models for NLP tasks. For example, in machine translation our input is a
sentence is some natural language and the output is the corresponding sentence in our target natural language; both have variable length. NLP isn't the only domain of interest here, RNNs
are general to any sequence. To put it in different terms, anytime when **memory** would
be useful, RNNs have shown to be a great model.

RNNs are also [Turing Complete](https://simple.wikipedia.org/wiki/Turing_complete), which means any program you write can be learned and therefore performed by an RNN. There's some preliminary results here with [Neural Turing Machines](http://arxiv.org/abs/1410.5401) inferring algorithms such as copying and sorting. Neural Turing Machines are quite a bit more sophisticated than RNNs but they both take advantage of memory, we won't cover them in this post.

## What does an RNN even look like?

Glad you asked. Like this:

![Picture of RNN](/images/rnn_part1/rnn.png)

$\theta$ represents the parameters of the RNN (weights and biases).

The circle is the RNN cell or layer, this is where the computation takes place. The computation varies
depending on the type of cell, we'll worry more about this when we get to the implementations. For now just know it puts your CPU/GPU to work.

The most important takeaway from the above image is the loop arrow and its interaction with the hidden state. On each computation on $\theta$ we take into consideration the input **and** the previous hidden state. If we *unroll* the RNN we can visualize the time aspect a bit better.

![RNN unrolled](/images/rnn_part1/rnn_unrolled.png)

I think this picture gets across the time interaction a bit better. My issue with this picture is
it's easy to think we now have multiple cells. One for time at t-1, t, t+1, etc. This is not the case.

Notice the `$\theta$` parameter is the same across the timesteps. This means we're **sharing** (not copying!) information across time, which is kind of the point. If you're familiar with pointers in programming lanaguages, it's like that. We're pointing to `$\theta$`. This simplication also dramatically lowers the memory cost.

Alright, now we get to the fun part, implementations!

### The classic RNN

The classic/vanilla RNN, unfortunately it doesn't taste as good as the ice cream :-(

<div>
$$
h_t = tanh(x_tU + h_{t-1}W)\\
o_t = h_tV
$$
</div>

* `$x_t$` is the input for the current timestep.
* `$o_t$` is the output for the current timestep to the next layer.
* `$h_t$`is the current hidden state, `$h_{t-1}$` the previous.
* `$U$`, `$W$`, `$V$` are the parameters we encapsulated as `$\theta$` earlier. This is what we're optimizing during our training.

Let's see how we translate this into code.

```python
# sizes matter right now
input_size = 10
hidden_size = 10

#
# rnn params
#
U = np.random.randn(input_size, hidden_size)
W = np.random.randn(hidden_size, hidden_size)
V = np.random.randn(hidden_size, hidden_size)

def rnn(xt, prev_h):
    ht = np.tanh(xt.dot(U) + prev_h.dot(W))
    ot = ht.dot(V)
    return ot, ht
```
It's simple already powerful for shorter sequences, but, it's lacking otherwise we wouldn't need LSTMs or GRUs. So what's the problem?

The problem is known as the [*vanishing gradient*](https://en.wikipedia.org/wiki/Vanishing_gradient_problem). Think about what happens when we multiply numbers less than 1 together. They get smaller at an exponential rate approaching 0.

How does this relate to RNNs?

1. With *unrolling* RNNs are essentially transformed to very deep feedforward networks. This means during backpropagation we're chaining several product operations.

2. Our activations functions are the sigmoid or tanh functions. These functions squish the outputs, either `$(0, 1)$` or `$(-1, 1)$` respectively. The derivatives are also small:

**Sigmoid derivative**

![Sigmoid derivative](/images/rnn_part1/sigmoid_derivative.gif)

**Tanh derivative**

![Tanh derivative](/images/rnn_part1/tanh_derivative.gif)

We see here 0.25 and 1 are the largest either derivative gets.

3. We initialize our weights to small numbers.

Putting all these pieces together we can see why the error gradient becomes 0, once this happens we stop learning. Ok, so make the weights larger, that'll do the trick right? Not quite, that
leads to the *exploding gradient* problem. Instead of going to 0, we go to `$\pm\infty$`.
In general our gradients are *unstable*.

Fragile little things these gradients are. It's like we're Jospeh Gordon-Levitt in [*The Walk*](https://en.wikipedia.org/wiki/The_Walk_\(2015_film\)), one false move and it's all over...

It's ok though, you can put a smile back on your face! LSTMs/GRUs are here to save
the day.

Some awesome resources for learning more about backprop and vanishing gradients:

* [How the backpropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html)
* [Calculus on Computational Graphs: Backpropagation](http://colah.github.io/posts/2015-08-Backprop/)
* [Vanishing gradients](http://neuralnetworksanddeeplearning.com/chap5.html#the_vanishing_gradient_problem)

### Long Short Term Memory (LSTM)

<div>$$
f_t = \sigma(xW_{xf} + h_{t-1}W_{hf} + b_f\\
i_t = \sigma(xW_{xi} + h_{t-1}W_{hi} + b_i\\
o_t = \sigma(xW_{xo} + h_{t-1}W_{ho} + b_o\\
\tilde C_t = tanh(xW_{xc} + h_{t-1}W_{hc} + b_c\\
C_t = f_t * C_{t-1} + i_t * \tilde C_t\\
h_t = o_t * tanh(C_t)
$$</div>

* `$f_t$` is the forget gate, determines how much of the previous memory state is remembered.
* `$i_t$` is the input gate, determines how much of the candidate memory state is used.
* `$o_t$` is the output gate, determines how much of of the current memory state is next to the next layer.
* `$C_t$` is the current memory state, `$C_{t-1}$` the previous and `$\tilde C_t$` the candidate for the current memory.

This an LSTM. Much more complicated than a vanilla RNN right? Let's break it down.

The most important part is this equation `$C_t = f_t * C_{t-1} + i_t * \tilde C_t$`. The current memory state is determined by the past `$f_t * C_{t-1}$` and the current input `$i_t * \tilde C_t$`. Remember sigmoid (`$\sigma$`) squishes the output to `$(0, 1)$`. If we get close to 1 we want to remember that bit, close to 0, forget about it!

The gates allow the LSTM to choose the values to send through. The combination of a memory state and gating circumvents the vanishing gradient problem we were plagued with earlier.

For more on LSTMs and pretty graphics see [this post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) by Christopher Olah.

Alright, onwards to the code.

```python
#
# lstm params
#

# forget gate
W_xf = np.random.randn(input_size, hidden_size)
W_hf = np.random.randn(hidden_size, hidden_size)
b_f = np.ones(hidden_size).reshape(-1, 1)

# input gate
W_xi = np.random.randn(input_size, hidden_size)
W_hi = np.random.randn(hidden_size, hidden_size)
b_i = np.ones(hidden_size).reshape(-1, 1)

# output gate
W_xo = np.random.randn(input_size, hidden_size)
W_ho = np.random.randn(hidden_size, hidden_size)
b_o = np.ones(hidden_size).reshape(-1, 1)

# candidate memory state
W_xc = np.random.randn(input_size, hidden_size)
W_hc = np.random.randn(hidden_size, hidden_size)
b_c = np.ones(hidden_size).reshape(-1, 1)

def lstm(xt, prev_c, prev_h):
	ft = sigmoid(xt.dot(W_xf) + prev_h.dot(W_hf) +  b_f)
	it = sigmoid(xt.dot(W_xi) + prev_h.dot(W_hi) +  b_i)
	ot = sigmoid(xt.dot(W_xo) + prev_h.dot(W_ho) +  b_o)

	candidate_memory = np.tanh(xt.dot(W_xc) + prev_h.dot(W_hc) +  b_c)
	ct = ft * prev_c + it * candidate_memory
	ht = ot * np.tanh(ct)

	# ct and ht are the outputs, we copy ht and send it to the
	# next layer
	return np.copy(ht), ct, ht
  ```

Some things to note:

1. We have many more parameters (not cool)
2. We have 3 inputs and 3 outputs.
3. The output for the next layer and next hidden state are the same.
4. We have biases, but we could have biases in a vanilla RNN as well. Biases are particularly more important in LSTMs, especially in the forget gate.

You might have noticed this, but an optimization we can do is combine all the gates and candidate memory into one matrix. We get 4 for the price of 1 this way.

Still, you might wonder if we can do better, do we really need all these parameters?

### Gated Recurrent Unit (GRU)

<div>$$
z_t = \sigma(xW_{xz} + h_{t-1}W_{hz}) \\
r_t = \sigma(xW_{xr} + h_{t-1}W_{hr}) \\
\tilde h_t = tanh(xW_{xc} + (h_{t-1} * r_t) W_{hc}) \\
h_t = (1 - z_t) * h_{t-1} + z_t * \tilde h_t
$$</div>

* `$z_t$` is the update gate, merged forget and input gates.
* `$r_t$` is the reset gate, when off it makes the GRU act as if
it's reading the first input of a sequence.
* `$\tilde h_t$` is the candidate activation, merged memory state
and hidden state.
* `$h_t$` is the hidden state.

Not as bad as an LSTM right? GRUs match the benefits of LSTMs but are conceptually simpler, as you can see. Because of this, they're gaining more and more traction.

```python
# update gate
W_xz = np.random.randn(input_size, hidden_size)
W_hz = np.random.randn(hidden_size, hidden_size)

# reset gate
W_xr = np.random.randn(input_size, hidden_size)
W_hr = np.random.randn(hidden_size, hidden_size)

# candidate activations
W_xc = np.random.randn(input_size, hidden_size)
W_hc = np.random.randn(hidden_size, hidden_size)

def gru(xt, prev_h):
	zt = sigmoid(xt.dot(W_xz) + prev_h.dot(W_hz))
	rt = sigmoid(xt.dot(W_xr) + prev_h.dot(W_hr))
	candidate_activation = np.tanh(xt.dot(W_xc) + (rt * prev_h).dot(W_hc))
	ht = (1. - zt) * prev_h + zt * candidate_activation

	# similar to a lstm we copy ht for the output
	# to the next layer
	return np.copy(ht), ht
```

We can optimize similarly to the LSTM by combining the update and reset gates.

Alright, we've covered a lot, but we've also skipped some details that are important
during training. We'll go over all that fun stuff in part 2. For now I just wanted
focus purely on the RNN and its variants. Once we have this down the rest
will be much easier!
