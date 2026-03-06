"""
Exercise 5: Training
=====================
A model with random weights is useless.  Training adjusts the weights
so that the model assigns high probability to real data and low
probability to nonsense.

You will:
  1. Compute the cross-entropy loss (how wrong is the model?)
  2. Implement SGD (the simplest optimizer)
  3. Implement Adam (a much better optimizer)
  4. Watch the loss decrease as the model learns

The most satisfying moment: seeing the loss drop from ~3.3 (random
guessing) to below 2.5 (the model is learning patterns in names).
"""

import os
import math
import random
random.seed(42)

# == PROVIDED: Value class + building blocks + GPT ==============
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1
    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._children: build(c)
                topo.append(v)
        build(self)
        self.grad = 1
        for v in reversed(topo):
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

# Model
n_embd = 16
n_head = 4
n_layer = 1
block_size = 16
vocab_size = 27
head_dim = n_embd // n_head

matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'wpe': matrix(block_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd),
}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = [p for mat in state_dict.values() for row in mat for p in row]

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)
    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]
    logits = linear(x, state_dict['lm_head'])
    return logits

# Dataset
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
random.shuffle(docs)
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
# == END PROVIDED ==============================================

print(f"Dataset: {len(docs)} names, {vocab_size} tokens, {len(params)} params")


# == THINK =====================================================
#
# The model has 27 possible output tokens and starts with random
# weights.  A random model assigns roughly equal probability to
# each token: p = 1/27.
#
# The cross-entropy loss for one prediction is -log(p_correct).
# What is -log(1/27)?   Calculate it now.
#
# This is your EXPECTED INITIAL LOSS.  If your first loss value
# is very different from this number, something is wrong.
# ==============================================================


# ==============================================================
# PART A: LOSS FUNCTION
# ==============================================================

# -- TODO 1: Compute cross-entropy loss for one document -------
#
# Given a tokenized document [BOS, t1, t2, ..., tn, BOS]:
#   - For each position, feed the current token into gpt()
#   - Get the predicted probability distribution (softmax of logits)
#   - The loss at that position is -log(prob[target_token])
#     where target_token is the NEXT token in the sequence
#   - Average the per-position losses
#
# Implement compute_loss(tokens) that returns the average loss
# as a Value (so we can call .backward() on it).

def compute_loss(tokens):
    """Compute average cross-entropy loss over a tokenized document.

    Args:
        tokens: list of int, e.g. [BOS, 7, 4, 11, 11, 14, BOS]
    Returns:
        loss: a Value (the average negative log-likelihood)
    """
    n = min(block_size, len(tokens) - 1)
    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]

    raise NotImplementedError("TODO 1: compute cross-entropy loss")


# ==============================================================
# PART B: OPTIMIZERS
# ==============================================================

# -- TODO 2: SGD (Stochastic Gradient Descent) -----------------
#
# The simplest optimizer.  After computing gradients:
#   p.data -= learning_rate * p.grad
#   p.grad = 0    (reset for next step)
#
# Implement sgd_step(params, learning_rate).

def sgd_step(params, learning_rate):
    raise NotImplementedError("TODO 2: SGD step")


# -- TODO 3: Adam optimizer ------------------------------------
#
# Adam tracks two running averages per parameter:
#   m  — momentum (smoothed gradient)
#   v  — RMS (smoothed squared gradient)
#
# Update rules (for each parameter i):
#   m[i] = beta1 * m[i] + (1 - beta1) * grad
#   v[i] = beta2 * v[i] + (1 - beta2) * grad^2
#   m_hat = m[i] / (1 - beta1^t)          (bias correction)
#   v_hat = v[i] / (1 - beta2^t)          (bias correction)
#   param -= lr * m_hat / (sqrt(v_hat) + eps)
#   grad = 0
#
# Implement adam_step(params, m_buf, v_buf, lr, step, beta1, beta2, eps).
# step is 1-indexed (first call is step=1).

def adam_step(params, m_buf, v_buf, lr, step, beta1=0.85, beta2=0.99, eps=1e-8):
    raise NotImplementedError("TODO 3: Adam step")


# ==============================================================
# TRAINING
# ==============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" Exercise 5: Training")
    print("=" * 60)

    # --- Phase 1: Train with SGD (50 steps) ---
    print("\n--- Phase 1: SGD (50 steps) ---")
    sgd_losses = []
    for step in range(50):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        loss = compute_loss(tokens)
        loss.backward()
        sgd_step(params, learning_rate=0.01)
        sgd_losses.append(loss.data)
        if step == 0 or (step + 1) % 10 == 0:
            print(f"  step {step+1:3d} | loss {loss.data:.4f}")

    # == Check your THINK answer ================================
    # Is the initial loss close to -log(1/27) = 3.296 ?
    # If it's much higher or lower, debug your compute_loss.
    # ==========================================================
    print(f"\n  Initial loss: {sgd_losses[0]:.4f}  (expected ~3.30)")

    # --- Phase 2: Re-initialize and train with Adam (200 steps) ---
    # (We re-init so both optimizers start from the same point)
    print("\n--- Phase 2: Adam (200 steps) ---")
    random.seed(42)
    state_dict['wte'] = matrix(vocab_size, n_embd)
    state_dict['wpe'] = matrix(block_size, n_embd)
    state_dict['lm_head'] = matrix(vocab_size, n_embd)
    for i in range(n_layer):
        state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
        state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
    params[:] = [p for mat in state_dict.values() for row in mat for p in row]
    random.shuffle(docs)

    m_buf = [0.0] * len(params)
    v_buf = [0.0] * len(params)
    num_steps = 200
    adam_losses = []

    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        loss = compute_loss(tokens)
        loss.backward()
        lr_t = 0.01 * (1 - step / num_steps)    # linear decay
        adam_step(params, m_buf, v_buf, lr_t, step + 1)
        adam_losses.append(loss.data)
        if step == 0 or (step + 1) % 20 == 0:
            print(f"  step {step+1:3d} | loss {loss.data:.4f}")

    # --- Verification ---
    print("\n--- Results ---")
    print(f"  SGD  after  50 steps: {sgd_losses[-1]:.4f}")
    print(f"  Adam after 200 steps: {adam_losses[-1]:.4f}")

    if adam_losses[-1] < adam_losses[0]:
        print("  [pass] loss decreased during training")
    else:
        print("  [FAIL] loss did not decrease — check your implementation")

    if adam_losses[-1] < 2.5:
        print("  [pass] final loss < 2.5 (good convergence)")
    else:
        print("  [note] final loss >= 2.5 — may need more steps")

    print("\n" + "=" * 60)
    print(" Exercise 5 complete!")
    print("=" * 60)


# == REFLECT ===================================================
#
# 1. Compare SGD and Adam after 50 steps each.  Which converges
#    faster?  Adam uses momentum (smoothed gradient direction) and
#    adaptive learning rates (different rate per parameter).
#    Why does this help?
#
# 2. We used linear learning rate decay: lr decreases to zero
#    over training.  Why not keep lr constant?  What happens
#    if lr is too high at the end of training?  (Think about
#    overshooting the minimum.)
#
# 3. The initial loss should be ~3.30.  After training, it drops
#    to ~2.x.  The theoretical minimum is 0 (perfect prediction).
#    Why can't this tiny model reach 0?  What would need to
#    change for it to get closer?
# ==============================================================

# == EXPLORE (optional) ========================================
#
# 1. Try learning_rate = 0.1  and  learning_rate = 0.001 with
#    Adam.  What happens in each case?  Plot the loss curves
#    if you can (matplotlib or just print them).
#
# 2. Try to OVERFIT on a single name: set docs = ["hello"]
#    and train for 1000 steps.  Can the model perfectly predict
#    "hello"?  What is the minimum achievable loss?
#    (Hint: think about the BOS token — what comes after BOS
#    is always "h", so the model can get that one right.
#    But what about the positions where multiple names in the
#    real dataset would diverge?)
#
# 3. The model trains on one document per step (batch_size=1).
#    Real LLMs use large batches (thousands of sequences).
#    Why?  What would change if we averaged the loss over 10
#    documents per step instead of 1?
# ==============================================================
