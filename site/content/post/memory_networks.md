+++
date = "2015-12-21"
title = "Memory Networks"
tags = ["Deep Learning", "AI"]
math = true
+++

Neural network approaches have traditionally lacked a memory component. By memory component I mean something akin to RAM in a computer. So, RNNs don't quite fit the picture as they provide a mechanism for learning long-term dependencies over time.

Explicit memory has become more popular as of late but the implementation we'll focus here is that of a [Memory Network](http://arxiv.org/abs/1410.3916). Actually an [End-To-End Memory Network](http://arxiv.org/abs/1503.08895). I'll describe the difference shortly, however, the general framework is similar.

A Memory Network can be described by 4 components:

1. Input feature map (I) converts the input x into a input feature representation.
2. Generalization (G) updates the current memory given the new input and previous memory. $m_i = G(m_i, I(x), m)$ for all i (slots of memory).
3. Output feature map (O) computes the output features $o = O(I(x), m)$
4. Response \(R) decode output features back to a representation for the user.

The main underlying idea is there's this memory $m$ which we update and use to make predictions. Think of it as short-term storage.

### Supervised vs Weakly Supervised

> I'll abbreviate Memory Network as MemNN and End-To-End Memory Networks as MemN2N from now on.

MemNN can be described as supervised and MemN2N as weakly supervised. The former means we explicitly say which memories to pick with some score function S. We usually pick the slot that gives the max value for S. In a weakly supervised setting we do a softmax over S. We want the same thing to happen but we're not being as bossy.


### Implementation

My implementation of MemN2N can be found [here](https://github.com/domluna/memn2n). I'm using the [bAbl tasks](https://research.facebook.com/researchers/1543934539189348) introduced in [this paper](http://arxiv.org/abs/1502.05698) for evaluation.

Ok, so let's break down the implementation of MemN2N in terms of the framework described above (I, G, O, R).

#### Input map (I)

This can get as complicated as you like, but, we keep it simple here and just assign an unique ID to each word in the vocabulary (words in test and training sets). We do this to both the story (sentences) and query. The answer is one-hot encoded.

If necessary with pad sentences with a nil word, the nil word's ID in this case is 0. Stories are also padded with empty memories (sentences filled with nil words).

#### Generalization (G)

Get embeddings from matrices $A$, $B$, $C$. The matrices are shaped (vocab_size, embedding_size). $A$ and $C$ are give embeddings for sentences and $B$ for the query.

Next elementwise multiply the embeddings by the **position encoding**. The position encoding (PE) allows the order of words to affect the memories.

```python
def _inference(self, stories, queries):
    with tf.name_scope("inference"):
        q_emb = tf.nn.embedding_lookup(self.B, queries)
        u_k = tf.reduce_sum(q_emb * self._encoding, 1)
        for _ in range(self._hops):
            i_emb = tf.nn.embedding_lookup(self.A, stories)
            o_emb = tf.nn.embedding_lookup(self.C, stories)
            # Memories
            m = tf.reduce_sum(i_emb * self._encoding, 2)
            c = tf.reduce_sum(o_emb * self._encoding, 2)
```

The number of hops are similar to numbers of layers in a traditional NN. The weights are shared over each hop.

We then do a dot product between each memory $m_i$ and the query representation $u$. A softmax over give us normalized probabilities representating importance of each memory.

<div>$$
p_i = {softmax(u^T{m_i})}
$$</div>

Two reasons why this part of code is so long:

1. No `tf.reduce_dot` operation and the first dimension the batch size so I have to do some
fancy manipulations.
2. Problem with empty memories (discussed below).

```python
def _input_module(self, m, u):
    with tf.name_scope("input_module"):
        # Currently tensorflow does not support reduce_dot, so this
        # is a little hack to get around that.
        u_temp = tf.transpose(tf.expand_dims(u, -1), [0, 2, 1])
        dotted = tf.reduce_sum(m_i * u_temp, 2)
        # Because we pad empty memories to conform to a memory_size
        # we add a large enough negative value such that the softmax
        # value of the empty memory is 0.
        # Otherwise, empty memories, depending on the memory_size will
        # have a larger and larger impact.
        bs = tf.shape(dotted)[0]
        tt = tf.fill(tf.pack([bs, self._memory_size]), -1000.0)
        cond = tf.not_equal(dotted, 0.0)
        # Returns softmax probabilities, acts as an attention mechanism
        # to signal the importance of memories.
        return tf.nn.softmax(tf.select(cond, dotted, tt))
```


There's a small hack here due to empty memories. If we have an empty memory then its
$p_i$ value is 0; as it should be. The issue is when softmax is applied we end up assigning empty memories a small probability, since exp(0) > 0. Therefore, we're giving importance to an empty memory. If we have many empty memories this becomes a problem. Assigning each empty $p_i$ to a largish negative number solves this issue.

We'll make use of the probabilities in the output map.

#### Output map (O)

<div>$$
o = \sum_i {p_i}{c_i}
$$</div>

```python
def _output_module(self, c, probs):
    with tf.name_scope("output_module"):
        probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
        c_temp = tf.transpose(c, [0, 2, 1])
        return tf.reduce_sum(c_temp * probs_temp, 2)
```

#### Response \(R)

$k$ represents the current hop.

<div>$$
u^{k+1} = ReLU({u^k} H + o^k)
$$</div>

```python
probs = self._input_module(m, u_k)
o_k = self._output_module(c, probs)
u_k = tf.matmul(u_k, self.H) + o_k
u_k = tf.nn.relu(u_k)
```

If it's the last hop

<div>$$
softmax(u^{k+1} W)
$$</div>

```python
tf.matmul(u_k, self.W)
```

This gives the probability distribution over the vocabulary. The prediction is the argmax.

### Evaluation

My previous results could only pass based on training accuracy, however with the addition of temporal encoding which gives importance to the order of memories
we now pass 5 tasks based on testing accuracy and have 80%+ testing accuracy on several others. Each of the 20 bAbI tasks were evaluated individually so there's reason to believe the results could be improved if a joint model were trained.

Passes:

1,4,12,15,20

Here's the setup:

* Adam with 0.01 learning rate
* 200 epochs
* batch size = 32
* 0.9/0.1 train/validation split
* 1k examples in train/test (2k total)
* Memory size is set to min(50, max\_story\_size)
* I also use [Gradient Noise](http://arxiv.org/abs/1511.06807) with a fixed standard deviation of 0.001. This was shown to be a good value for their Memory Network test.

Another notable mention is I removed the matrix $C$ discussed above. This appears to be done in some form in later papers where the vocabulary size is much larger. In my experiments removing $C$ and using $A$ in its replacement improves results. I don't know exactly why this is, but I speculate due to the small dataset if both $A$ and $C$ are used neither generalizes well enough.

### Improvements & Future Directions

* In the current model we only predict 1 word. We could make this variadic by using an RNN in the output.
* Regularization. Currently lacking regularization. Dropout and Batch Normalization would be probably be good here. I tried L2 loss but had issues with weird shape errors from Tensorflow.
* Try different memory models. This includes both representations and how the actual memory itself functions. For example, currently we reset the memory on each example but it might be interesting only remove the amount needed for the current input. Kind of like a sliding window.
* Predict words not in the original vocab. In the original paper this is done by an n-grams approach
where we generate the embedding from the context of the word, as well as the word itself.
* Experiment with more complex computations ( conv layers, rnn layers, variational layers, etc )
* Reinforcement Learning


