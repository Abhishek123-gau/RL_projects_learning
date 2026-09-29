# Actor-Critic: Understanding Random Sampling and Weight Updates

## The key confusion

A common question in Actor-Critic / REINFORCE is:

> If the action is chosen randomly, how can that random action tell us how to update the weights?

The important idea is:

**We do not update the weights because the action was random.**

We update the weights based on **what happened after that action**.

---
## Think of one timestep

Suppose the policy says:

text
P(RIGHT) = 0.7
P(LEFT)  = 0.3

Then sampling happens:

0.7 probability → RIGHT
0.3 probability → LEFT

Suppose this particular time randomness gave:

RIGHT

Now the environment runs RIGHT and gives:

reward + next state

The Critic evaluates what happened and we get:

TD error = +5

That means:

"The randomly chosen RIGHT action turned out better than the Critic expected."

So we reinforce that choice.

RIGHT was sampled
       ↓
outcome was GOOD
       ↓
increase probability of RIGHT

If instead the Critic gives:

TD error = -5

then:

RIGHT was sampled
       ↓
outcome was BAD
       ↓
decrease probability of RIGHT
The important point

We are not saying:

"RIGHT was randomly selected, therefore increase its probability."

We are saying:

"RIGHT was selected, and after trying it, the outcome was better than expected. Therefore, increase its probability."

And if the outcome was worse than expected:

"RIGHT was selected, but the outcome was worse than expected. Therefore, decrease its probability."

So the complete idea is:

Policy
  ↓
P(RIGHT) = 0.7
P(LEFT)  = 0.3
  ↓
Random sampling
  ↓
RIGHT
  ↓
Environment
  ↓
Reward + next state
  ↓
Critic
  ↓
TD error
  ↓
┌──────────────────────────┐
│ TD error > 0             │
│ outcome better than      │
│ expected                 │
│        ↓                 │
│ reinforce chosen action  │
└──────────────────────────┘

OR

┌──────────────────────────┐
│ TD error < 0             │
│ outcome worse than       │
│ expected                 │
│        ↓                 │
│ weaken chosen action     │
└──────────────────────────┘



So in our current implementation:

reward = 1

but mathematically:

reward = whatever the environment says



# changes can be done
1. Mkae the critic the action lr differnet
2. Mkae the poilcy better as everything is linear
3. Mkae the other functions in RL like defined reward it's poor just decide to have reward 1 everytime the physcis more depends on angle and force that can be taken as reward or chnaged or variable
