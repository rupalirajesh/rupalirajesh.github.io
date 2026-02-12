---
layout: single
title: "Training a Language Model"
date: 2026-02-12
author: Rupali Rajesh
mathjax: true
---

# Overall Training Pipeline

Training an LLM involves the following steps -

  - ***Pre-training*** the model where we train it on a large amount of unlabeled data.

  - ***Post-training*** the model for alignment. To use and query the LLM, we need to align its behaviour to follow a particular style.
      - Supervised Fine-Tuning (***SFT***): The model is trained on a labeled dataset of human-written or synthetically generated examples of ideal prompts + responses.
      - Preference Fine-Tuning (PreFT): The model is aligned to human preferences using either ***RLHF*** or ***DPO***.

![](/assets/images/IMG_0793.jpg){: .align-center style="width: 70%; max-width: 600px;"}

# Pretraining

Pretraining is the stage where we build the foundation model by feeding
large and diverse text from sources such as Wikipedia, books, code
repositories like GitHub, research articles, news websites, etc.
(essentially any publicly available internet text) to the model.

The goal is to have the model learn the general structure of language,
facts about the world, code syntax, etc.

During pretraining, the model sees raw text as a continuous sequence of
tokens and learns to predict the next token at each position. This is
known as ***next-token prediction***. So, though we do not provide ideal
prompt-response examples, the correct next token is already in the text.
The labels come from the data itself. Therefore, we call this
self-supervised learning.

During supervised fine-tuning (SFT), the data is organized into ideal
prompt-response (instruction-completion) pairs $(x,y)$. The model is
trained to generate the completion $y$ conditioned on the instruction
$x$. In contrast, during pretraining, the model learns to model entire
text streams autoregressively, without distinguishing between
instruction and completion. The underlying next-token prediction
mechanism is the same in both cases and is explained in more detail in
the SFT section.

# Why alignment?

LLMs are pre-trained on a large amount of data, for example the entirety
of Wikipedia or the whole internet. The base LLM is simply trained on
raw text sequences without an explicit prompt-response structure. We
need to align its behaviour to follow a particular style of answering
questions.

For example, if we prompt our model as follows: "Explain
gradient descent.", it might imitate a book, forum debate, or the
relevant research paper. We would like it to condense information and
answer questions directly, in this case.

Here are some examples of guidelines we would like our LLM to follow.
  - Avoid offensive language.
  - Do not answer questions that might encourage racism or violence (for example, we might want to avoid giving users information on how to build bombs or procure weapons).
  - Answer questions cordially, briefly, without too much technical information, in sections, or in an essay-style.

SFT teaches the model how to respond. Preference Optimization teaches
the model which responses are better.

# Supervised Fine Tuning (SFT)

***Instruction tuning*** is where we adapt the base model to our desired
style of input. To do this, we might feed the model chat templates in
question-answer format. A good example of this data might be stack
exchange posts or reddit posts, which are naturally in this format.

We generate *labeled data* where each example is of the
form$\lbrack instruction (or prompt), completion (or ideal response)\rbrack.$

For explanation models, this might be
$\lbrack instruction = input + prediction, completion = trusted explanation\rbrack$.

This is our dataset of "good responses" for the model to
learn from.

All frontier models use *synthetic data* along with human-written
labeled prompts. Synthetic data is generated as follows - from high
quality human prompts, we create modified versions of the instructions
using a strong language model, then generate completions with another
(or the same) strong language model to generate more training data.

The model is fine-tuned with ***cross-entropy loss***.

(Skip this next part if you already know how next token prediction and
this optimization works.)

Let's represent the dataset as follows:
$D = \{(x^{(i)},y^{(i)}{\}^{N}}_{i = 1}$.

Here, $x^{(i)}$ is the instruction, $y^{(i)}$ is the completion, and $N$
is the number of training examples.

The model does not predict the whole completion at once. The response is
divided into tokens (potentially words or sub-words). Models predict the
next token in a sequence, given prior context.

Mathematically, if a completion has tokens
$y^{(i)} = ({y_{1}}^{(i)},{y_{2}}^{(i)},...,{y_{T_{i}}}^{(i)})$, then
the model defines
$p(y^{(i)}|x^{(i)}) = \prod_{j = 1}^{T_{i}}{}p({y_{j}}^{(i)}|x^{(i)},{y_{< j}}^{(i)})$.
This is the product of the probability of predicting the first token,
then the second token given the correct first token, and then the third
token given the correct first two tokens, and so on. This is the *chain
rule*.

We want the model to assign the highest probability to the ideal
response. So we define a *maximum likelihood objective*:
$max\prod_{i = 1}^{N}{}p(y^{(i)}|x^{(i)})$ which is the same as
maximising
$max\sum_{i = 1}^{N}{}logp(y^{(i)}|x^{(i)}) = max\sum_{i = 1}^{N}{}\sum_{j = 1}^{T_{i}}{}logp({y_{j}}^{(i)}|x^{(i)},{y_{< j}}^{(i)})$.

Training frameworks like to minimize functions, so we convert this to a
negative log-likelihood loss function given by:
$L_{SFT} = - \sum_{i = 1}^{N}{}\sum_{j = 1}^{T_{i}}{}logp({y_{j}}^{(i)}|x^{(i)},{y_{< j}}^{(i)})$.

Minimizing this function is the same as minimizing this function divided
by a constant N, so we redefine the loss function as:
$L_{SFT} = - \frac{1}{N}\sum_{i = 1}^{N}{}\sum_{j = 1}^{T_{i}}{}logp({y_{j}}^{(i)}|x^{(i)},{y_{< j}}^{(i)}) = E_{(x,y)\sim D}(\sum_{j = 1}^{T_{i}}{} - logp(y_{j}|x^{},{y_{< j}}^{}))$.

Therefore, the final SFT objective is to minimize
$L_{SFT} = E_{(x,y)\sim D}(\sum_{j = 1}^{T_{i}}{} - logp(y_{j}|x^{},{y_{< j}}^{}))$.

In SFT, the cross entropy loss is exactly the negative log-likelihood of
the ground truth next token, since the true distribution is one hot.

For a true distribution q(.) and the model distribution p(.), the cross
entropy is $H(p,q) = - \sum_{v}^{}{}q(v)logp(v)$. This is a weighted
average of each of the log probabilities.

At each given time step or token position j, we know the next true
token. Therefore, we have a one-hot distribution (among the discrete
possibilities in the vocabulary, only one of them is correct).
Mathematically, $q(v) = 1ifv = y_{j}andq(v) = 0otherwise.$

Now the cross entropy loss becomes
$H(p,q) = - \sum_{v}^{}{}q(v)logp(v|x,y_{< j}) = - logp(y_{j}|x,y_{< j})$

Summing over all of the tokens,
$H(p,q) = \sum_{j=1}^{T_i} - \log p\big(y_j^{(i)} \mid x, y_{<j}^{(i)}\big)$.
Averaging this over all of the examples in the dataset, we get exactly
our maximum likelihood objective.

Some small things to note are that -
  - The loss is only computed at the completion tokens.
  - The prompt tokens are used as context but are not penalized.
  - This tells the model what good explanations look like, anchors its behaviour, and sets the baseline for preference optimization to build on.

# Preference Optimization

Preference optimization is a post-training method where a model is
trained to prefer better responses over worse ones. This step aligns our
model to human preferences. We train it for chat *style*. It continues
building capabilities learned from SFT, but with lower absolute
magnitude of improvements. Why?

  - SFT does most of the heavy-lifting. So the model already understands the task, knows the skill, and produces reasonable outputs.
  - Preference signals are weaker than supervised labels. Getting something like "Response A is better than Response B" produces much noisier gradients with smaller step sizes than "Response C is the ideal response" which gives us target tokens.
  - Most preference methods penalize deviating from the SFT model, as we
will see.

This is a good thing. We do not want to stray too far away from our
original model to prevent reward hacking. We might accidentally train
the model to cater too much to human preferences. For example, in our
explanation model example, humans tend to rate longer explanations as
more trustworthy. Our model might learn this from the preference signals
and start generating unnecessarily long responses to get a higher
reward.

Preferences can come from humans (trained using RLHF) or strong models
(trained using RLAIF or UltraFeedback). We simply adjust the model such
that $p(y^{+}|x) > p(y^{-}|x)$. Here, p is the probability of the model
generating response y to an input x. There are several methods to do
this. Here, I discuss two famous techniques.

## Reinforcement Learning with Human Feedback (RLHF)

The first step is to ***collect preference data***. Let us denote the
SFT model as $\pi_{SFT}$. We sample multiple responses from $\pi_{SFT}$
and have human beings or a strong language model rank them. This gives
us data of the form $(x,y^{+},y^{-})withy^{+} \succ y^{-}$.

Next, we ***train a reward model***. We simply want the reward model to
represent these preferences we collected, in other words, we want our
reward function r to return $r(x,y^{+}) > r(x,y^{-})$.

How best to define and train this reward model is still pretty
unresearched. The reward model is usually initialized from the old SFT
model $\pi_{SFT}$. We derive the following loss function from the
Bradley-Terry model of preference,
$L_{r} = - log\sigma(r(x,y^{+}) - r(x,y^{-}))$.

The derivation is given below.

Given a pair of items i and j chosen from a population, the
Bradley-Terry model estimates that the comparison $i > j$ has the
following probability of being true:
$P(i > j) = \frac{p_{i}}{p_{i} + p_{j}}$ where $p_{i}$ is a positive
real-valued score assigned to i. Bradley and Terry themselves defined
exponential score functions $p_{i} = e^{\beta_{i}}$ so that:
$P(i > j) = \frac{e^{\beta_{i}}}{e^{\beta_{i}} + e^{\beta_{j}}} = \frac{1}{1 + e^{- (\beta_{i} - \beta_{j})}} = \sigma(\beta_{i} - \beta_{j})$.
In the Bradley-Terry model we know the functional form and try to infer
the parameters $\beta_{i}$ whereas in logistic regression, we know the
parameters and try to infer the functional form of $P(i > j)$.

For our preference learning task, $r(x,y)$ is a real-valued score and a
higher r indicates a higher preference. So, we simply set
$\beta_{i} = r(x,y_{i})$. Therefore,
$P(y^{+} > y^{-}) = \sigma(r(x,y^{+}) - r(x,y^{-}))$. We would like to
maximise this probability for every such preference pair, giving us the
loss function: $L_{r} = - log\sigma(r(x,y^{+}) - r(x,y^{-}))$.

We now treat the LLM as an RL agent:

State: $x,y_{< t}$

Action: $y_{t}$

Episode reward: $r(x,y$)

We define the following objective:
$maxE_{y\sim\pi_{\theta}}\lbrack r(x,y) - \beta KL(\pi_{\theta}(y|x)||\pi_{SFT}(y|x))\rbrack$.

This maximises the reward while the KL distance term keeps the behaviour
of our optimized model similar to the SFT model. This prevents reward
hacking and ensures stability.

This objective function is optimized using an ***RL algorithm*** like
***PPO*** or ***GRPO***. I explain how PPO works below as an excuse to
learn about it. Feel free to skip!

![](/assets/images/IMG_0792.jpg){: .align-center style="width: 70%; max-width: 600px;"}

### Proximal Policy Optimization (PPO)

PPO is an RL algorithm that improves a policy without letting it deviate
too much from the original policy, using a *clipped objective.*

In policy gradient methods for RL, we update the parameters $\theta$ to increase the expected reward:

![](/assets/images/eq1.png){: .align-center style="width: 20%; max-width: 600px;"}

Here, $\theta$ are the model parameters (the transformer weights), $\pi_{\theta}$ is the language model’s output distribution, $\tau$ is a fully generated response, and $R(\tau)$ is the total reward for that response. The expectation averages rewards over many responses the model could generate.

Let us quickly frame the RL problem before seeing how to solve this -

Stochastic policy: $\pi_{\theta}(a \mid s)$

State: $s = (x,y_{< t})$

Action: $a = y_{t}$

Reward: $r$ given at the end of the sequence.

As an example -

![](/assets/images/IMG_0794-2.jpg){: .align-center style="width: 70%; max-width: 600px;"}

To maximise this expectation, we use policy gradient updates to the
parameters. Here is the vanilla policy gradient.

Objective:

![](/assets/images/eq3.png){: .align-center style="width: 20%; max-width: 600px;"}

Update: $\theta_{new} = \theta_{old} + \alpha\nabla_{\theta}(E)$

This has several issues. While it is a simple and unbiased gradient,
there is high variance, the learning rate is fragile, large updates
collapse the policy while small updates learn too slowly. There is also
no constraint on how much the policy changes from the old one.

Define probability ratio:
$r_{\theta}(s,a) = \frac{\pi_{\theta}(a|s)}{\pi_{\theta_{old}}(a|s)}$.
We can interpret it in the following way: if $r > 1$, the new policy
likes this action more. Advantage is defined as
$A(s,a) = Q(s,a) - V(s)$, where Q tells us the expected total future
reward if you are in state s, take action a, and follow the policy and V
is the expected total future reward if you are in state and follow the
same policy. If you can guess the relationship, V is the average over
all actions, and Q is for the specific action. So advantage is
intuitively the reward we get by following action a in state s versus
the average reward in that state. Advantage is often assigned to all
tokens in the response. The reward is sequence-level.

Modern RL uses two models -
  1. ***Actor*** or the policy $\pi_{\theta}(a|s)$ which decides what action
to take. In LLMs, this asks what token the model should output next.
  2. ***Critic*** or the value function $V(s)$ which estimates how good the
current situation is by evaluating the actor's decisions.
Without this, the policy gradients would be the vanilla policy
gradients. The critic allows us to compute advantage which answers if
this action is better or worse than expected, stabilizing training.
  The critic is trained by regression. $\overset{\hat{}}{V}(s_{t}) = r_{t} + \gamma V(s_{t + 1})$. The loss is typically simply MSE, $L_{value} = E\lbrack(V(s_{t}) - \overset{\hat{}}{V}(s_{t}))^{2}\rbrack$.

Let's rewrite the objective using old policy data:
$J(\theta) = E_{\theta_{old}}\lbrack r_{\theta}(s,a)A(s,a)\rbrack.$

So an earlier solution was Trust Region Policy Optimization (TRPO). TRPO
constrains the new objective using KL divergence:
$KL(\pi_{new}||\pi_{old}) \leq \varepsilon.$ However, to solve this we
need second-order methods (take my word for it) and other stable but
costly methods.

Using PPO, we clip the objective, which allows us to enforce almost the
same constraint with a first-order stochastic gradient descent.
$L_{policy}(\theta) = E\lbrack min(r(\theta)A,clip(r(\theta),1 - \epsilon,1 + \epsilon)A)\rbrack$
where the clip function is defined as follows:

![](/assets/images/eq2.png){: .align-center style="width: 20%; max-width: 600px;"}

It constrains $r(\theta)$ to lie between $1 - \epsilon$ and
$1 + \epsilon$. Intuitively, this means $1 + \epsilon$ is the maximum
allowed increase in probability and similarly, $1 - \epsilon$ is the
maximum allowed decrease. We cannot change the probability of an action
by more than $\sim\epsilon$ in one update. This prevents large policy
shifts.

In practice, PPO optimizes three terms:
$L = L_{policy} + c_{1}L_{value} - c_{2}L_{entropy}$. It updates the
actor using the clipped objective and advantage estimates, and trains
the critic to better estimate $V(s)$ which improves future advantage
estimates. The entropy bonus encourages exploration $H(\pi_{\theta})$.

Now, let's look at LLM-specific PPO in RLHF.

There is no environment reward, only a learned reward model that scores
completions generated by the model: $r_{RM}(x,y)$ (RM is the reward
model). The reward is zero except at the final token.

At each step, here is the defined reward:
$r = r_{RM} - KL(\pi_{\theta}||\pi_{SFT})$. More
practically,![](/assets/images/eq4.png){: .align-center style="width: 20%; max-width: 600px;"}.
The computer returns are
$\overset{\hat{}}{R_{t}} = \sum_{i = t}^{T}{}\gamma^{i - t}r_{i}$.

$V(s_{t})$ is learned by the critic. Advantage
$A_{t} = \overset{\hat{}}{R_{t}} - V(s_{t})$. Everything else stays
exactly the same.

In summary, the reward model tells us what humans prefer, the value
function stabilizes learning, the KL term prevents large deviations from
SFT, and the clip prevents violent updates.

#### What is wrong with PPO for LLM alignment? 

  - Reward models are brittle. They are trained on comparisons, extrapolate
poorly, and are easy to hack. Since PPO tries to find actions that
maximize the reward (it is not trying to imitate data), it is prone to
reward hacking. This is amplified by the critic and propagates the
reward model's mistake across states.

  - The critic is expensive and unstable. Training PPO requires an actor,
critic, reward model, and reference model. Most of the PPO instability
comes from the critic, not the policy. This is because critic overfits.

  - Rollouts are full prompt to response trajectories. These are sampled
from the current policy, then scored by the reward model, and used to
compute advantages for updating the policy and critic. For this, we must
repeatedly generate fresh on-policy data (from $\pi_{\theta}$ currently
being updated), which is extremely expensive for LLMs.

  - It is incompatible with our data. Human feedback is pairwise preferences
and offline, however PPO wants scalar rewards and online interactions.
Training reward models and converting preferences to scalar rewards are
hacks that add noise and increase the risk of failure.

  - The KL penalty coefficient $\beta$ must be tuned, which is expensive.

## Direct Preference Optimization (DPO)

This method allows us to optimize the policy directly from preference
data without a reward model, critic, rollouts, or KL tuning.

DPO tries to optimize directly from human preference data of the form
$(x,y^{+},y^{-})$ where $y^{+}$ is the preferred response to input $x$
and $y^{-}$ is the less preferred response to input $x$.

Let us assume the reward model is of the form:
$$r(x,y) = \beta(\log\pi_{ideal}(y|x) - \log\pi_{SFT}(y|x))$$ where $\pi_{ideal}$ is
the ideal human-preferred policy (simply the theoretical model that
gives $$\pi_{ideal}(y^{\text{+}}|x) > \pi_{ideal}(y^{\text{-}}|x)$$). It is a stochastic policy,
so better answers are more likely and worse ones are less likely).

Let's derive this. This is known as an inverse RL
derivation since we infer the reward model from observed behaviour.

We want to optimize
$maxE_{y\sim\pi}\lbrack r(x,y)\rbrack - \beta KL(\pi||\pi_{SFT})$.
$KL(\pi||\pi_{SFT}) = \sum_{y}^{}{}\pi(y|x)(log\pi(y|x) - log\pi_{SFT}(y|x))$.
Therefore the objective becomes
$\sum_{y}^{}{}\pi(y|x)\lbrack r(x,y) - \beta log\pi(y|x) + \beta log\pi_{SFT}(y|x)\rbrack$.
A policy must satisfy $\sum_{y}^{}{}\pi(y|x) = 1$.

Therefore, the Lagrangian becomes
$L(\pi,\lambda) = \sum_{y}^{}{}\pi(y)\lbrack r(x,y) - \beta log\pi(y) + \beta log\pi_{SFT}(y)\rbrack + \lambda(1 - \sum_{y}^{}{}\pi(y))$.
The x is there, just dropped for now to make it more readable.

We optimize w.r.t. the policy $\pi(y)$ by taking derivative and setting
to zero:
$\frac{\partial L}{\partial\pi(y)} = r(x,y) - \beta(log\pi(y) + 1) + \beta log\pi_{SFT}(y) - \lambda = 0.$

This gives:

$$log\pi(y) = \frac{1}{\beta}(r(x,y) + \beta log\pi_{SFT}(y) - \lambda - \beta)$$

$$\pi(y) = exp(\frac{r(x,y)}{\beta})\pi_{SFT}(y)exp(\frac{- \lambda - \beta}{\beta})$$

The last exponential term is a normalization constant $Z^{- 1}$, which
ensures $\sum_{y}^{}{}\pi^{*}(y|x) = 1$ so it is a valid probability
distribution.

So we get the following solution:
$\pi^{*}(y|x) = \frac{1}{Z(x)}\pi_{SFT}(y|x)exp(\frac{r(x,y)}{\beta})$.

Taking logs:
$\log \pi_{ideal}(y \mid x) = \log \pi_{SFT}(y \mid x) + \frac{1}{\beta} r(x,y) - \log Z(x)$.

Rearranging the terms, we get:

$$r(x,y) = \beta \big( \log \pi_{ideal}(y \mid x) - \log \pi_{SFT}(y \mid x) \big) + \beta \log Z(x)$$

![](/assets/images/eq5.png){: .align-center style="width: 20%; max-width: 600px;"}

Modeling preference probability using Bradley-Terry derived earlier, we
have:

$$P(y^{+} \succ y^{-}|x) = \sigma(r(y^{+}|x) - r(y^{-}|x))$$

$$P(y^{+} \succ y^{-}|x) = \sigma(\beta\lbrack log\pi_{ideal}(y^{+}|x) - log\pi_{ideal}(y^{-}|x) - (log\pi_{SFT}(y^{+}|x) - log\pi_{SFT}(y^{-}|x))\rbrack)$$

The SFT terms are constant with respect to $\theta$ and can be removed
since they do not change the gradient. We also replace the ideal policy
by our trainable policy $\pi_{\theta}$.

$P(y^{+} \succ y^{-}|x) = \sigma(\beta\lbrack log\pi_{\theta}(y^{+}|x) - log\pi_{\theta}(y^{-}|x)\rbrack)$.

For each preference pair, we minimize the following DPO loss:
$L_{DPO} = - log\sigma(\beta\lbrack log\pi_{\theta}(y^{+}|x) - log\pi_{\theta}(y^{-}|x)\rbrack)$.
This loss pushes the model to increase and decrease probability of
$y^{+}$ and $y^{-}$ respectively, relative to the SFT model. So, DPO
training is simply logistic regression on preferences.

Some key properties are -
  - Training is offline. The loss online depends on a static dataset of
preferences and $log\pi_{\theta}(y^{\pm},x)$. We do not need new
rollouts (samples generated from $\pi_{\theta}$).
  - The closeness to SFT enforcement is implicit. Our inverse RL derivation
assumes that $\pi^{*} \propto \pi_{SFT}exp(r/\beta)$. So when we
optimize preferences, we enforce closeness to SFT.
  - Gradients are bounded. Let's call
$\Delta = \beta\lbrack log\pi_{\theta}(y^{+}|x) - log\pi_{\theta}(y^{-}|x)\rbrack$.
Therefore,
$\frac{\partial L_{DPO}}{\partial\Delta} = \sigma(\Delta) - 1\epsilon\lbrack - 1,0\rbrack$.
Therefore, there are no exploding gradients. In PPO, when $\pi_{SFT}$ is
small, the importance ratio $r(\theta) = \frac{\pi_{\theta}}{\pi_{SFT}}$
explodes giving us a huge gradient.
