"""
Exercise 3: Self-Attention
===========================
The attention mechanism is the heart of the Transformer.  It lets
each token look at every previous token and decide which ones are
relevant to predicting what comes next.

The idea:  each token produces a Query ("what am I looking for?"),
a Key ("what do I contain?"), and a Value ("what information do I
carry").  Attention scores are computed by comparing each Query
against all available Keys, then using those scores to take a
weighted sum of Values.

You will implement a single attention head.  In Exercise 4, you
will split into multiple heads.
"""

import math
import random
random.seed(42)

# == PROVIDED: Value class + building blocks ====================
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
# == END PROVIDED ==============================================


# Model dimensions
n_embd = 16
n_head = 1           # single head for this exercise
head_dim = n_embd    # with 1 head, head_dim = n_embd
block_size = 16
vocab_size = 27

# Random parameters
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'wpe': matrix(block_size, n_embd),
    'attn_wq': matrix(n_embd, n_embd),
    'attn_wk': matrix(n_embd, n_embd),
    'attn_wv': matrix(n_embd, n_embd),
    'attn_wo': matrix(n_embd, n_embd),
}


# == THINK =====================================================
#
# This model is AUTOREGRESSIVE: when processing token at position t,
# it can only attend to tokens at positions 0, 1, ..., t (itself
# and the past).  It cannot see the future.
#
# 1. When processing the FIRST token (position 0), how many
#    tokens can it attend to?  1, unica posicao
# 
# What will the attention weights be?
#.  Os pesos de atenção serão [1.0] — softmax de um único valor dá sempre 1.
#
# 2. When processing the THIRD token (position 2), it attends to
#    3 tokens.  The attention weights will sum to 1.  But are
#    they equal?  What determines how much weight each gets?

#    Não são iguais. Cada peso depende da similaridade (Q·K) entre o query
#    do token atual e a key de cada token passado. maior similaridade = mais peso.
#
# Write your predictions.  Test 4 will check prediction #1.
# ==============================================================


def single_head_attn(token_id, pos_id, keys, values):
    """
    Process one token through single-head attention.

    Args:
        token_id:  integer token ID
        pos_id:    integer position in the sequence
        keys:      list of key vectors seen so far (grows each call)
        values:    list of value vectors seen so far (grows each call)

    Returns:
        output:       list of Values, length n_embd
        attn_weights: list of Values (the attention distribution)
    """
    # PROVIDED: embedding lookup
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    # -- TODO 1: Compute Q, K, V projections --------------------
    #
    # Use linear() to project x through the weight matrices
    # 'attn_wq', 'attn_wk', 'attn_wv' to get vectors q, k, v.
    #
    # Then APPEND k to the keys list and v to the values list.
    # This builds the "KV cache" — the memory of past tokens.
    q = linear(x, state_dict['attn_wq'])
    k = linear(x, state_dict['attn_wk'])
    v = linear(x, state_dict['attn_wv'])
    keys.append(k)
    values.append(v)

    # -- TODO 2: Compute attention scores -----------------------
    #
    # For each past position t (0 to len(keys)-1), compute:
    #   score_t = (q . keys[t]) / sqrt(head_dim)
    #
    # where q . keys[t] is the dot product across all dimensions.
    # The sqrt(head_dim) scaling prevents scores from growing too
    # large as the dimension increases.
    #
    # Result: a list of Values, one score per position.
    scale = head_dim ** 0.5
    attn_scores = []
    for t in range(len(keys)):
        score = 0
        for j in range(head_dim):
            score += q[j] * keys[t][j]
        attn_scores.append(score / scale)

    # -- TODO 3: Attention weights ------------------------------
    #
    # Apply softmax to the attention scores to get a probability
    # distribution over positions.
    attn_weights = softmax(attn_scores)

    # -- TODO 4: Weighted sum of values -------------------------
    #
    # Compute the output as a weighted sum of value vectors:
    #   output_j = sum_t ( attn_weights[t] * values[t][j] )
    #
    # for each dimension j in range(head_dim).
    # Result: a list of Values, length head_dim.
    head_out = []
    for j in range(head_dim):
        total = 0
        for t in range(len(values)):
            total += attn_weights[t] * values[t][j]
        head_out.append(total)

    # PROVIDED: output projection
    output = linear(head_out, state_dict['attn_wo'])
    return output, attn_weights


# ==============================================================
# TESTS
# ==============================================================
if __name__ == "__main__":
    print("=" * 60)
    print(" Exercise 3: Self-Attention")
    print("=" * 60)
    print()

    # Test 1: output dimension
    keys, values = [], []
    output, weights = single_head_attn(0, 0, keys, values)
    assert len(output) == n_embd, \
        f"output should have {n_embd} elements, got {len(output)}"
    print("  [pass] output has correct dimension")

    # Test 2: attention weights sum to 1
    total = sum(w.data for w in weights)
    assert abs(total - 1.0) < 1e-6, \
        f"attention weights should sum to 1.0, got {total}"
    print("  [pass] attention weights sum to 1.0")

    # Test 3: KV cache grows correctly
    keys, values = [], []
    all_weights = []
    for pos in range(3):
        output, weights = single_head_attn(pos % vocab_size, pos, keys, values)
        all_weights.append(weights)
    assert len(all_weights[0]) == 1, "1st token attends to 1 position"
    assert len(all_weights[1]) == 2, "2nd token attends to 2 positions"
    assert len(all_weights[2]) == 3, "3rd token attends to 3 positions"
    print("  [pass] KV cache grows: 1, 2, 3 positions")

    # Test 4: First token — check your prediction
    first_weight = all_weights[0][0].data
    assert abs(first_weight - 1.0) < 1e-6, \
        f"first token should attend 100% to itself, got {first_weight:.4f}"
    print("  [pass] first token: 100% attention on itself (as predicted?)")

    # Test 5: Different inputs produce different outputs
    keys1, vals1 = [], []
    out1, _ = single_head_attn(0, 0, keys1, vals1)
    keys2, vals2 = [], []
    out2, _ = single_head_attn(5, 0, keys2, vals2)
    differ = any(abs(a.data - b.data) > 1e-10 for a, b in zip(out1, out2))
    assert differ, "different tokens should produce different outputs"
    print("  [pass] different tokens produce different outputs")

    print("\n" + "=" * 60)
    print(" Exercise 3 complete!")
    print("=" * 60)


# == REFLECT ===================================================
#
# 1. The first token always attends 100% to itself because there
#    is nothing else to attend to.  But later tokens distribute
#    attention across all past tokens.  What determines HOW the
#    attention is distributed?  (Hint: it's the Q-K similarity.)
#
# 2. We scale by sqrt(head_dim).  What would happen without
#    this scaling?  Think about what large dot products do to
#    the softmax distribution.  (Hint: softmax saturates.)
#
# 3. This is CAUSAL (autoregressive) attention: each token sees
#    only the past.  BERT uses BIDIRECTIONAL attention where
#    each token sees all other tokens.  When would you want
#    each type?
# ==============================================================

# == EXPLORE (optional) ========================================
#
# 1. Remove the sqrt(head_dim) scaling.  Process a 5-token
#    sequence and print the attention weights at each position.
#    Are they more "peaked" (concentrated on one token) or more
#    "uniform" without the scaling?  Why?
#
# 2. Currently we use SELF-attention: Q, K, V all come from the
#    same input x.  In CROSS-attention, Q comes from one sequence
#    and K, V from another (e.g., in machine translation).
#    Modify single_head_attn to accept a separate source sequence
#    for K and V.
# ==============================================================
