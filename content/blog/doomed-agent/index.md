---
title: The Case of the Doomed Agent
date: "2016-07-11"
---

> This project was done as the Capstone for the Udacity Machine Learning Nanodegree

### Definition

#### Project Overview

[OpenAI Gym](https://gym.openai.com/) is a toolkit for developing and comparing reinforcement learning algorithms. It supports teaching agents everything from walking to playing games like [Pong](https://gym.openai.com/envs/Pong-ram-v0) or [Go](https://gym.openai.com/envs/Go19x19-v0). In this project we will by training Reinforcement Learning (RL) agents to solve environments based on the Doom video game. Specifically, the [DoomCorridor-v0](https://gym.openai.com/envs/DoomCorridor-v0) and [DoomHealthGathering-v0](https://gym.openai.com/envs/DoomHealthGathering-v0) environments. Either environment is considered solved when the agent averages an episode reward >= 1000 over 100 consecutive episodes. An episode being a sequence (s0, a1, s1, a2, s2, …., st),  s0 is the initial state, st is the terminal state and (action, state) pairs in between. A reward is given at the end of every action, the episode reward is the summation of these individual rewards.


#### Problem Statement


The objective of the agent in DoomCorridor-v0 is to reach the vest at the end of the corridor as fast as possible without dying. There are 6 enemies (3 groups of 2) that can possibly kill the agent. The input data (observations) we use for training the agent are the frames (pixels) and action space is discrete, consisting of 6 actions.

The objective of the agent in DoomHealthGathering-v0 is to stay alive for as long as possible by collecting health packs. The ground is poison so simply standing on it results in a loss of health. The input data (observations) we use for training the agent are the frames (pixels) and action space is discrete, consisting of 3 actions.

So how do we train the agent?

The initial input is provided by the `reset` method on the environment. 

```
initial_state = env.reset()
```

The following inputs will be provided by calling the `step` method. 

```
next_state, reward, done, info = env.step(action)
```

`next_state` and `reward` are self explanatory, `done` entails whether the agent has reached a terminal state and info is miscellaneous information that will be ignored for the purposes of this project. 

The following details one timestep:

1. Receive input from environment by calling either `reset` or `step` as described above. The input is an image representing the current state.
2. Preprocess input.
3. Run inference on the input with a neural network, the output will be a valid action.
4. Postprocess action so it can be feed back into the environment.


The above will be repeated until a terminal state or timestep limit is reached. During this process we accumulate the states, rewards, actions, etc as they will factor into the later training procedure. This sequence of state, action pairings is known as a **trajectory**.

Once we’ve collected a number of trajectories (dependent on the batch size) we’ll calculate our loss and run our SGD variant training the neural network. This will count as a single iteration.

For completeness, here’s the full algorithm in pseudocode. $N$ is the number of iteration, $M$ is the batch size.

```
for i in 1, …, N
	trajectories = []
	for j in 1, …, M
		Sample trajectory t_j and add t_j to trajectories
	Compute loss given trajectories
	Optimize neural network parameters w.r.t loss
```

#### Metrics

An environment is considered solved if the agent averages greater or equal to an episodic reward of 1000 over 100 consecutive episodes. Because of this, the **reward** will be the main metric used.

### Analysis

#### Data Exploration

As mentioned previously, we use the observation space as input data for the RL agent. The observation comes in the form of 3D image data (width, height, channels). The Doom environment allows us to choose from a range of screen resolutions (width, height) options. We choose the smallest available resolution (160, 120) for performance reasons. Thus, the observation space is a 3D Tensor of shape (160, 120, 3).

#### Data Visualization

This video shows the DoomHealthGathering environment.

<iframe width="420" height="315" src="https://www.youtube.com/embed/re6hkcTWVUY" frameborder="0" allowfullscreen></iframe>

#### Algorithms and Techniques

The algorithms used in this project will be based on deep RL, the intersection of deep and reinforcement learning. Reinforcement learning is concerned with an agent and an environment. The goal of the agent is to act optimally in the environment according to an objective. This is done by learning an expectation of future rewards from a state. 

$$
F(state) = E[R]\\
R = \sum_{i=k}^N \gamma^{i-k} r_i\\
$$

Where $\gamma$ is the discount value $[0, 1]$, that determines how much long-term rewards are valued. We pick the action that maximizes $F(state)$ value.

There are two prevailing approaches in representing $F(state, action)$ or $F(state)$.

* Tabular - We store in memory (state, action) pairs mapping to a value.
* Function approximation - We store parameters in memory. These parameters are trained using stochastic gradient descent with a machine learning model such as neural networks.

There are pros and cons to both approaches but function approximation was chosen and it comes down to two factors.

1. Our state space is exponential, without any preprocessing we have 57600 floating point numbers representing the state, even with the preprocessing we’ll discuss later we’re still left with 300 floating point numbers. Even if we only allowed 10 values per number, that’s still $10^{300}$ possible states making a tabular approach intractable!
2. There have been recent advances in both deep and reinforcement learning that make reinforcement learning through function approximation feasible.


We settle on [Policy Gradient](https://gym.openai.com/docs/rl#policy-gradients) methods and neural networks for our approach. Policy gradients are used because they directly optimize for the cumulative reward (unlike [Q-Learning](https://gym.openai.com/docs/rl#q-learning) and can be applied straightforwardly with nonlinear approximators such as neural networks. Policy gradient methods have been known for instability due to high variance gradient estimates and therefore impractical. However, with [**Trust Region Policy Optimization**](https://arxiv.org/abs/1502.05477) (TRPO) being introduced, policy gradient methods have since shown success in learning difficult problems.

So why does TRPO help?

A neural network with parameters represents a function space or manifold, the function output being dependent on the parameters. The idea of the **Natural Gradient** is to traverse this manifold to the find the optimal function for the task. This approach is appealing because the parameterization of the network does not matter. For example, even though the gradients of `tanh` and `relu` activations are different, since they’re part of the same function space the activation choice shouldn't affect the optimization procedure. Given this, it’s evident how using the natural gradient over vanilla backpropagation would be desirable. TRPO builds on the natural gradient by providing additional constraints for the gradient step size and direction updates.

During the training process we actually train two separate neural networks simultaneously, called the **policy** and the **baseline** networks respectively. Both consist the same architecture until the final layer, the difference being the policy calculates action probabilities and thus the final layer is a softmax, while the baseline outputs a scalar, the final layer being linear with a single output. The scalar produced by the baseline is an estimate of the future reward given a state. Given the baseline value and actual rewards from a trajectory, we can calculate what’s known as the **advantage**.

The advantage is calculated by subtracting the baseline value from the sum of rewards (described above). The intuition being if the advantage is > 0, the sampled trajectory is a profitable one so the agent should be encouraged to follow similar ones. Conversely, if the advantage < 0, the sampled trajectory is non-profitable and similar trajectories should be discouraged. The advantage helps reduce the high variance in gradient estimates.

#### Benchmark

For our benchmark we will use the condition for solving the environment, that is, averaging an episode reward >= 1000 over 100 consecutive episodes.

### Methodology

#### Data Preprocessing

We preprocesses both the observation space and the action space. The observation space is initially of shape (160, 120, 3). We transform it to (15, 20, 1) by a grayscale and resize operation, note this is now read as (height, width, channels). We do this transformation for performance reasons, for example instead of 160 * 120 * 3 = 57600 features per observation we have 20 * 15 = 300 features. That’s ~0.05% of the original size!

An action is represented as an array of 43 elements, each element representing an type of action. Simultaneous actions are supported but for simplicity we pick a single action at each timestep. For the DoomCorridor-v0 environment we only have 6 possible actions.

array index -> name

* 0 -> ATTACK
* 10 -> MOVE\_RIGHT
* 11 -> MOVE\_LEFT
* 13 -> MOVE\_FORWARD
* 14 -> TURN\_RIGHT
* 15 -> TURN\_LEFT

The output of our neural network will be an integer 0-5, a softmax over 6 actions.  The  integer is mapped to one of the above action indexes and the element value is set to 1. We then pass the array in the environment as our action. The mapping for DoomHealthGathering-v0 is similar.

#### Implementation

We build upon [Modular RL](https://github.com/joschu/modular_rl), an implementation of TRPO using Keras and Theano. 

Our contributions:

* `agentzoo.py` - add support for CNN TRPO agents
* `filtered_env.py` - add skiprate, action filters, support for Doom envs
* `run_ff.py`, `run_cnn.py` - do preprocessing with filters, implement loading snapshots



Skiprate & Preprocessing/Filters

During training a **skiprate** is when we predict an action to use that action for the the next k timesteps, the benefit being we explore states k times faster, thereby encouraging exploration. This also makes sense intuitively, think about talking a walk in a park. Chances are you don’t rethink where you should be going every step.

```
def _step(self, ac):
    nac = self.act_filter(ac) if self.act_filter else ac
    if self.skiprate:
        total_nrew = 0.0
        total_rew = 0.0
        num_steps = np.random.randint(self.skiprate[0], self.skiprate[1])
        nob = None
        done = False
        for _ in range(num_steps):
            ob, rew, done, info = self.env.step(nac)
            nob = self.ob_filter(ob) if self.ob_filter else ob
            nrew = self.rew_filter(rew) if self.rew_filter else rew
            total_nrew += nrew
            total_rew += rew
            if done:
                info["reward_raw"] = total_rew
                return (nob, total_nrew, done, info)
        info["reward_raw"] = total_rew
        return (nob, total_nrew, done, info)
    else:
        ob, rew, done, info = self.env.step(nac)
        nob = self.ob_filter(ob) if self.ob_filter else ob
        nrew = self.rew_filter(rew) if self.rew_filter else rew
        info["reward_raw"] = rew
        return (nob, nrew, done, info)
```

Above is the implementation of the skiprate along with the use of action and observation filters (we don’t use a reward filter).

The original action is processed such that we can call `env.step(...)` later.

```
nac = self.act_filter(ac) if self.act_filter else ac
```

Here we also see the state being processed from (160, 120, 3) to (15, 20, 1).

```
ob, rew, done, info = self.env.step(nac)
nob = self.ob_filter(ob) if self.ob_filter else ob
```

If we have a skiprate, every step we pick a number uniformly in our range.

```
num_steps = np.random.randint(self.skiprate[0], self.skiprate[1])
``` 

For each of these steps we repeat the same action and accumulate the reward, if during a step we reach a terminal state then the function will return early.

```
for _ in range(num_steps):
    ob, rew, done, info = self.env.step(nac)
    nob = self.ob_filter(ob) if self.ob_filter else ob
    nrew = self.rew_filter(rew) if self.rew_filter else rew
    total_nrew += nrew
    total_rew += rew
    if done:
        info["reward_raw"] = total_rew
        return (nob, total_nrew, done, info)
info["reward_raw"] = total_rew
return (nob, total_nrew, done, info)
```

Models & Agents

```
class TrpoAgent(AgentWithPolicy):
    options = MLP_OPTIONS + PG_OPTIONS + TrpoUpdater.options + FILTER_OPTIONS

    def __init__(self, ob_space, ac_space, usercfg):
        cfg = update_default_config(self.options, usercfg)
        ** policy, self.baseline = make_mlps(ob_space, ac_space, cfg) **
        obfilter, rewfilter = make_filters(cfg, ob_space)
        self.updater = TrpoUpdater(policy, cfg)
        AgentWithPolicy.__init__(self, policy, obfilter, rewfilter)
```


```
class TrpoAgentCNN(AgentWithPolicy):
    options = MLP_OPTIONS + PG_OPTIONS + TrpoUpdater.options + FILTER_OPTIONS

    def __init__(self, ob_space, ac_space, usercfg):
        cfg = update_default_config(self.options, usercfg)
        ** policy, self.baseline = make_cnn(ob_space, ac_space, cfg) **
        obfilter, rewfilter = make_filters(cfg, ob_space)
        self.updater = TrpoUpdater(policy, cfg)
        AgentWithPolicy.__init__(self, policy, obfilter, rewfilter)
```

The code for the agents is the same except for the construction of the neural network. This obviously differs due to creating a feedforward or convolutional network.

The final piece to the puzzle is `run_policy_gradient_algorithm`.

```
def run_policy_gradient_algorithm(env, agent, usercfg=None, callback=None):
    cfg = update_default_config(PG_OPTIONS, usercfg)
    cfg.update(usercfg)
    print "policy gradient config", cfg

    if cfg["parallel"]:
        raise NotImplementedError

    tstart = time.time()
    seed_iter = itertools.count()

    for _ in xrange(cfg["n_iter"]):
        # Rollouts ========
        paths = get_paths(env, agent, cfg, seed_iter)
        compute_advantage(agent.baseline, paths, gamma=cfg["gamma"], lam=cfg["lam"])
        # VF Update ========
        vf_stats = agent.baseline.fit(paths)
        # Pol Update ========
        pol_stats = agent.updater(paths)
        # Stats ========
        stats = OrderedDict()
        add_episode_stats(stats, paths)
        add_prefixed_stats(stats, "vf", vf_stats)
        add_prefixed_stats(stats, "pol", pol_stats)
        stats["TimeElapsed"] = time.time() - tstart
        if callback: callback(stats)
```

The most important snippet being

```
for _ in xrange(cfg["n_iter"]):
    # Rollouts ========
    paths = get_paths(env, agent, cfg, seed_iter)
    compute_advantage(agent.baseline, paths, gamma=cfg["gamma"], lam=cfg["lam"])
    # VF Update ========
    vf_stats = agent.baseline.fit(paths)
    # Pol Update ========
    pol_stats = agent.updater(paths)
```

`get_paths` returns a list of episodes, each episode contains information about the episode such as the reward at each timestep. Once we have the episodes we calculate the advantage and update the baseline and policy parameters.


#### Refinement

Initially we had no skiprate and used the default resolution of 640x480. This proved to be very costly in time so the skiprate and 160x120 resolution were introduced. I also experimented with another policy gradient method called [A3C](https://arxiv.org/abs/1602.01783), an async Actor-Critic method. I don’t talk much about A3C because I couldn’t get consistent results with it. This might be a fault of my implementation but even in the A3C paper the authors note they ran **each experiment 50 times and picked the 5 best results**. So it may just be A3C is unstable.

### Results

#### Model Evaluation and Validation

Hyperparameters

* `gamma` = 0.995, Discount factor
* `lam` = 0.97, Lambda parameter from [Generalized Advantage Estimation](http://arxiv.org/abs/1506.02438)
* `max_kl` = 0.01, Add multiple of the identity to the Fisher Matrix during optimization.
* `cg_dampling` = 0.1, KL-divergence between old and new policy (averaged over state-space).
* `activation` = tanh for feedforward, relu for CNN
* `n_iter` = 250 or manual stop
* `timesteps_per_batch` = 5000

Architectures

Feedforward Model

* layer 1: linear layer, 64 hidden neurons, tanh activation
* layer 2: linear layer, 64 hidden neurons, tanh activation

CNN Model

* layer 1: convolutional layer, 16 filters (4x4) stride 2, relu activation
* layer 2: convolutional layer, 16 filters (4x4) stride 2, relu activation
* layer 3: linear layer, 20 hidden neurons, relu activation

Both architectures are followed by a softmax layer for action probabilities.

The baselines follow the same structure as the policy networks except the softmax output layer is switched with a linear layer with 1 output value.

**DoomCorridor-v0**

[Feedforward](https://gym.openai.com/evaluations/eval_sSZnwzT2RTW6xwJm9hYIug)

The model converges to a solution that averages ~2270 reward/episode over the last ~5k episodes. This is far beyond the benchmark of averaging 1000 reward/episode over consecutive 100 episodes. 

[CNN](https://gym.openai.com/evaluations/eval_KotIO6YbTPyBeSN90MSNA)

Similar to the feedforward agent except convergence is quicker. This could be due to the convolutional network providing a better representation of the data.

**DoomHealthGathering-v0**

[Feedforward](https://gym.openai.com/evaluations/eval_WdQ8Gkk0S0SmKpxm4xY6PA)

The agent takes longer to converge than the DoomCorridor-v0 task, this isn’t surprising since the task is more difficult. After almost 10k episodes the environment is solved. 

[CNN](https://gym.openai.com/evaluations/eval_qfKxrgc7QPuyaTJ3DSywXw)

The agent never converges to a solution and appears to get stuck in a local minima.

The HealthGathering environment is interesting because the TRPO implementation used does not follow an exploration scheme such as linear decay (epsilon slowly decreasing over $t$ timesteps), so exploration isn’t explicitly encouraged. This might not matter if the agent was left to explore indefinately or over an extended period of time, but, since the agent will die shortly if it does not receive a health pack, the agent might never properly learn the reward associated with receiving a health pack. Due to this, it’s a possibility the CNN performs poorly simply due to the random placement of health packs in the environment.

### Conclusion

#### Free-Form Visualization

The videos below are played in reverse chronological order, the first being the final recorded episode and the last being the first recorded episode.

<video src="https://openai-kubernetes-prod-scoreboard.s3.amazonaws.com/v1/evaluations/eval_sSZnwzT2RTW6xwJm9hYIug/training_episode_batch_video.mp4"
poster="https://openai-kubernetes-prod-scoreboard.s3.amazonaws.com/v1/evaluations/eval_KotIO6YbTPyBeSN90MSNA/training_episode_batch_video_poster.jpg" controls height="270" width="360">
</video>

<video src="https://openai-kubernetes-prod-scoreboard.s3.amazonaws.com/v1/evaluations/eval_KotIO6YbTPyBeSN90MSNA/training_episode_batch_video.mp4"
poster="https://openai-kubernetes-prod-scoreboard.s3.amazonaws.com/v1/evaluations/eval_KotIO6YbTPyBeSN90MSNA/training_episode_batch_video_poster.jpg" controls height="270" width="360">
</video>

In both the above videos we can see the agent figures out the best strategy is to run directly to the vest and not concern itself with the enemies.

<video src="https://openai-kubernetes-prod-scoreboard.s3.amazonaws.com/v1/evaluations/eval_9fGQ2UmLSyq0uU5O7uEzQ/training_episode_batch_video.mp4"
poster="https://openai-kubernetes-prod-scoreboard.s3.amazonaws.com/v1/evaluations/eval_9fGQ2UmLSyq0uU5O7uEzQ/training_episode_batch_video_poster.jpg" controls height="270" width="360">
</video>

The agent learns it needs to pick up health packs in order to survive. We can also see the agent becoming less timid in its decision making as the episodes progress. Another interesting note is the agent doesn't appear to be completely convinced where the health pack is until it's very close to it. We can see this in the video but also by running `play.py` for several episodes with the saved snapshot. This shows high variance in the episode reward depending where the agent starts relative to the health packs.


<video src="https://openai-kubernetes-prod-scoreboard.s3.amazonaws.com/v1/evaluations/eval_qfKxrgc7QPuyaTJ3DSywXw/training_episode_batch_video.mp4"
poster="https://openai-kubernetes-prod-scoreboard.s3.amazonaws.com/v1/evaluations/eval_qfKxrgc7QPuyaTJ3DSywXw/training_episode_batch_video_poster.jpg" controls height="270" width="360">
</video>

The agent never converges to a solution. The agent was ran with multiple seeds to make sure a bad initialization wasn't at fault.

#### Reflection

In this project we trained reinforcement learning agents on the DoomCorridor-v0 and DoomHealthGathering-v0 environments provided as part of the OpenAI Gym toolkit. We modeled the agent as a neural network, implementing both feedforward and convolutional network architectures. The agent was trained using TRPO, an algorithm from a family of methods called Policy Gradients, which optimize to directly maximize the cumulative episodic reward. We successfully solve both environments as the agent receives an episodic reward of 1000 or greater over 100 consecutive episodes.

It’s Interesting how well the feedforward model performs. When it comes to Machine Learning tasks involving images convolutional networks have dramatically outperformed feedforward networks. So why does the feedforward model hold its own here? I hypothesize it’s due to the complexity of the input space. 

In more traditional image processing tasks the images come from the real world. We can see images from the real world carry much more complexity than the images provided by the Doom environments. In fact, most images from our environment look very similar, there’s not much variance. Because of this the feedforward network can learn a representation of the input space where it otherwise couldn’t.

Due to TRPO being empirically and theoretically sound going into the project I thought it would perform well, especially since I found good results running the algorithm on a variety of other environments part of the Gym toolkit. It’s astonishing how much more robust TRPO is over [REINFORCE](https://webdocs.cs.ualberta.ca/~sutton/williams-92.pdf) even though they are quite similar, the main difference being the use of the natural gradient.

#### Improvement

There are several directions we can follow which may improve our solution.

* Trying more neural network architectures and environments would be interesting. In particular [RNNs](https://en.wikipedia.org/wiki/Recurrent_neural_network) might prove valuable in environments where reward is relatively sparse, such as DoomHealthGathering-v0. 
* Different algorithms could be tried as well, such as Q-Learning or A3C. Even [evolutionary algorithms](https://en.wikipedia.org/wiki/Evolutionary_algorithm), outside the realm of traditional RL could prove interesting. 
* Improvements on just the algorithm itself. Whether it be incorporating different [exploration policies](http://arxiv.org/abs/1602.04621v3) or adding an [external memory](http://arxiv.org/abs/1606.04460v1).
* Different state representations such as the difference of the current and previous state or previous and current state tuples.

For the Corridor environment it seems the the optimal solution was found. However, the HealthGathering environment can most certainly be improved, as it’s far more open ended. Perhaps an entirely new benchmark for that environment can proposed, such as consistently staying alive for $t$ timesteps.