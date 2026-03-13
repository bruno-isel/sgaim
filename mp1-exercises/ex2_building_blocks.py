"""
Exercise 2: Neural Network Primitives
=======================================
Three small functions that appear everywhere inside a Transformer:

  linear   — matrix-vector multiplication (every layer uses this)
  softmax  — turns raw scores into a probability distribution
  rmsnorm  — normalizes vectors to stabilize training

Each is short, but each hides a subtlety worth understanding.
"""

import math

# == PROVIDED: Value class (your autograd engine from Ex1) ======
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
# == END PROVIDED ==============================================


# -- TODO 1: Linear (matrix-vector multiply) -------------------
#
# Inputs:
#   x — a list of Values, length n_in    (a vector)
#   w — a list of lists of Values, shape [n_out][n_in]  (a matrix)
#
# Output:
#   a list of Values, length n_out
#
# Each output element is the dot product of one row of w with x.

def linear(x, w):
    out = []
    for row in w:
        dot = sum(w_i * x_i for w_i, x_i in zip(row, x))
        out.append(dot)
    return out

# == THINK =====================================================
# softmax must produce a valid probability distribution.
# What two properties MUST the output satisfy?
# (Write them down before implementing.)
# ==============================================================

# -- TODO 2: Softmax -------------------------------------------
#
# Formula:   softmax(x_i) = exp(x_i) / sum_j exp(x_j)
#
# Input:   a list of Values (the "logits")
# Output:  a list of Values that form a probability distribution

def softmax(logits):
    raise NotImplementedError


# -- TODO 3: RMS Normalization ---------------------------------
#
# RMSNorm scales a vector so that its root-mean-square equals 1.
#
# Formula:   rms = sqrt( mean(x_i^2) )
#            output_i = x_i / rms
#
# Use a small epsilon (1e-5) to avoid division by zero:
#            output_i = x_i / sqrt( mean(x_i^2) + 1e-5 )

def rmsnorm(x):
    raise NotImplementedError


# ==============================================================
# TESTS
# ==============================================================
if __name__ == "__main__":
    import random
    random.seed(42)
    print("=" * 60)
    print(" Exercise 2: Neural Network Primitives")
    print("=" * 60)
    print()

    # --- linear ---
    x = [Value(1.0), Value(2.0), Value(3.0)]
    w = [[Value(0.1), Value(0.2), Value(0.3)],
         [Value(0.4), Value(0.5), Value(0.6)]]
    out = linear(x, w)
    assert len(out) == 2, f"linear output should have 2 elements, got {len(out)}"
    # 0.1*1 + 0.2*2 + 0.3*3 = 1.4
    assert abs(out[0].data - 1.4) < 1e-10
    # 0.4*1 + 0.5*2 + 0.6*3 = 3.2
    assert abs(out[1].data - 3.2) < 1e-10
    print("  [pass] linear: correct output shape and values")

    # --- softmax: basic properties ---
    logits = [Value(1.0), Value(2.0), Value(3.0)]
    probs = softmax(logits)
    total = sum(p.data for p in probs)
    assert abs(total - 1.0) < 1e-6, f"softmax should sum to 1.0, got {total}"
    assert all(p.data > 0 for p in probs), "softmax outputs must be positive"
    # Largest logit -> largest probability
    assert probs[2].data > probs[1].data > probs[0].data
    print("  [pass] softmax: sums to 1, all positive, order preserved")

    # --- softmax: numerical stability ---
    # == If this test fails with an overflow or nan: ============
    # Your softmax can't handle large inputs.
    # Think: what is exp(1000)?  Can your computer represent it?
    #
    # Mathematical fact that will save you:
    #   softmax(x) = softmax(x - c)   for ANY constant c.
    # This is because exp(x-c)/sum(exp(x-c)) = exp(x)/sum(exp(x))
    # when you factor out exp(-c) from numerator and denominator.
    #
    # How can you exploit this to prevent overflow?
    # ==========================================================
    logits_big = [Value(1000.0), Value(0.0), Value(0.0)]
    probs_big = softmax(logits_big)
    assert abs(probs_big[0].data - 1.0) < 1e-6, \
        f"softmax([1000,0,0])[0] should be ~1.0, got {probs_big[0].data}"
    print("  [pass] softmax: handles large values (numerical stability)")

    # --- rmsnorm ---
    x = [Value(3.0), Value(4.0), Value(5.0)]
    normed = rmsnorm(x)
    rms = (sum(v.data**2 for v in normed) / len(normed)) ** 0.5
    assert abs(rms - 1.0) < 0.01, f"rmsnorm output RMS should be ~1.0, got {rms}"
    print("  [pass] rmsnorm: output has unit RMS")

    # --- gradients flow through all three ---
    x = [Value(1.0), Value(2.0)]
    w = [[Value(0.5), Value(0.5)]]
    out = linear(x, w)
    out[0].backward()
    assert x[0].grad != 0, "gradient must flow through linear"

    logits = [Value(1.0), Value(2.0)]
    probs = softmax(logits)
    probs[0].log().backward()
    assert logits[0].grad != 0, "gradient must flow through softmax"

    x = [Value(3.0), Value(4.0)]
    normed = rmsnorm(x)
    normed[0].backward()
    assert x[0].grad != 0, "gradient must flow through rmsnorm"
    print("  [pass] gradients flow through all three functions")

    print("\n" + "=" * 60)
    print(" Exercise 2 complete!")
    print("=" * 60)


# == REFLECT ===================================================
# If you had to fix softmax for large inputs:
#   - Why does subtracting the max work mathematically?
#   - Would subtracting any other constant also work?
#   - Is this trick specific to softmax, or a general pattern
#     for working with exponentials?
# ==============================================================

# == EXPLORE (optional) ========================================
#
# 1. Implement LayerNorm instead of RMSNorm.
#    LayerNorm subtracts the mean AND divides by the std dev:
#      mu = mean(x)
#      sigma = sqrt(mean((x - mu)^2) + eps)
#      output = (x - mu) / sigma
#    What extra information does LayerNorm use that RMSNorm ignores?
#    When might that matter?
#
# 2. What does softmax do when ALL inputs are equal?
#    What about when one input is much larger than the rest?
#    Describe the two extremes and what controls where you land
#    between them.  (Hint: this relates to "temperature", which
#    you will meet in Exercise 6.)
#
# 3. Linear layers are the most parameter-hungry part of a
#    Transformer.  If x has dimension d and w has shape [4d, d],
#    how many parameters does this single linear layer have?
#    For GPT-3's d=12288, how many parameters is that?
# ==============================================================
