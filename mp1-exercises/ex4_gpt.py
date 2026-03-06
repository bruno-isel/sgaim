"""
Exercise 4: The GPT Model
===========================
Time to assemble the full model.  In Exercise 3, you built a single
attention head.  Now you will:

  1. Split attention into MULTIPLE heads (each sees a slice of the
     embedding, bringing different "perspectives")
  2. Add an MLP block (two linear layers with ReLU between them)
  3. Wire in residual connections (the input is added back to the
     output of each block, creating a "gradient highway")
  4. Put it all together: embeddings -> layers -> output logits

By the end, you will have a complete GPT forward pass — the same
architecture (at toy scale) behind GPT-2, GPT-3, and GPT-4.
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


# Model configuration
n_embd = 16
n_head = 4
n_layer = 1
block_size = 16
vocab_size = 27
head_dim = n_embd // n_head   # = 4

# Initialize all parameters
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


# == THINK =====================================================
#
# With n_embd=16 and n_head=4, each head sees only 4 dimensions
# out of 16.  Each head has its own Q, K, V "view" of the data.
#
# 1. Why split into multiple heads instead of one big head?
#    What might different heads learn to focus on?
#    (Analogy: reading a sentence for grammar vs. for meaning
#    vs. for emotional tone — different "perspectives".)
#
# 2. What is a residual connection?  If the input to a block is x
#    and the block computes f(x), the output with a residual is
#    x + f(x).   Why does this help training?
#    (Hint: what is the gradient of x + f(x) with respect to x?
#    Compare to the gradient of f(x) alone.)
# ==============================================================


def gpt(token_id, pos_id, keys, values):
    """
    Full GPT forward pass for a single token.

    Args:
        token_id: integer token ID
        pos_id:   integer position in the sequence
        keys:     list of n_layer lists, each accumulating key vectors
        values:   list of n_layer lists, each accumulating value vectors

    Returns:
        logits: list of Values, length vocab_size
    """

    # Step 1: Embedding
    # Look up the token embedding and position embedding,
    # add them together, then apply rmsnorm.
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):

        # ======================================================
        # Step 2a: Multi-head attention block
        # ======================================================
        x_residual = x
        x = rmsnorm(x)

        # Project to Q, K, V (full dimension)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)

        # -- TODO 1: Multi-head attention -----------------------
        #
        # Split Q, K, V into n_head heads, each of size head_dim.
        # For head h, the slice is [h*head_dim : (h+1)*head_dim].
        #
        # For each head:
        #   - Extract q_h (from q), k_h (from each key in cache),
        #     v_h (from each value in cache) for this head's slice
        #   - Compute attention: scores, softmax, weighted sum
        #     (same logic as Exercise 3, but on head_dim dimensions)
        #   - Collect the head's output (head_dim values)
        #
        # Concatenate all head outputs into a single list (x_attn)
        # of length n_embd.
        #
        # Then project: x = linear(x_attn, attn_wo)
        raise NotImplementedError("TODO 1: multi-head attention")

        # -- TODO 2: Residual connection around attention -------
        #
        # Add the original input (x_residual) back to x.
        # This is x = x + x_residual, element-wise.
        raise NotImplementedError("TODO 2: residual connection")

        # ======================================================
        # Step 2b: MLP block
        # ======================================================
        x_residual = x
        x = rmsnorm(x)

        # -- TODO 3: MLP (feed-forward network) ----------------
        #
        # Two linear layers with ReLU activation in between:
        #   x = linear(x, mlp_fc1)    [n_embd -> 4*n_embd]
        #   x = relu(x)               [element-wise]
        #   x = linear(x, mlp_fc2)    [4*n_embd -> n_embd]
        #
        # The "4x expansion" is standard in Transformers — the MLP
        # first expands the representation, then compresses it back.
        raise NotImplementedError("TODO 3: MLP block")

        # -- TODO 4: Residual connection around MLP -------------
        raise NotImplementedError("TODO 4: residual connection")

    # Step 3: Project to vocabulary
    logits = linear(x, state_dict['lm_head'])
    return logits


# ==============================================================
# TESTS
# ==============================================================
if __name__ == "__main__":
    print("=" * 60)
    print(" Exercise 4: The GPT Model")
    print("=" * 60)
    print()

    # Test 1: output shape
    kc = [[] for _ in range(n_layer)]
    vc = [[] for _ in range(n_layer)]
    logits = gpt(0, 0, kc, vc)
    assert len(logits) == vocab_size, \
        f"output should have {vocab_size} logits, got {len(logits)}"
    print("  [pass] output has vocab_size (27) logits")

    # Test 2: multi-token sequence
    kc = [[] for _ in range(n_layer)]
    vc = [[] for _ in range(n_layer)]
    for pos in range(3):
        logits = gpt(pos, pos, kc, vc)
    assert len(logits) == vocab_size
    print("  [pass] 3-token sequence processes correctly")

    # Test 3: different inputs produce different outputs
    kc1, vc1 = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    out1 = gpt(0, 0, kc1, vc1)
    kc2, vc2 = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    out2 = gpt(5, 0, kc2, vc2)
    differ = any(abs(a.data - b.data) > 1e-10 for a, b in zip(out1, out2))
    assert differ, "different tokens must produce different outputs"
    print("  [pass] different inputs -> different outputs")

    # Test 4: parameter count
    assert len(params) == 4192, f"expected 4192 params, got {len(params)}"
    print("  [pass] parameter count = 4,192")

    # == REFLECT ================================================
    # Count the parameters by component:
    #   - wte:     27 * 16     =  432   (token embeddings)
    #   - wpe:     16 * 16     =  256   (position embeddings)
    #   - attn:    4 * 16 * 16 = 1024   (Q, K, V, O projections)
    #   - mlp:     16*64 + 64*16 = 2048 (fc1 + fc2)
    #   - lm_head: 27 * 16     =  432   (output projection)
    #   - Total:                  4192
    #
    # Notice: the MLP has 2048 params, attention has 1024.
    # The MLP is TWICE as large as attention!  This ratio holds
    # in real GPT models too.  Most parameters live in the MLP.
    # ==========================================================

    # Test 5: gradients flow end-to-end
    kc = [[] for _ in range(n_layer)]
    vc = [[] for _ in range(n_layer)]
    logits = gpt(0, 0, kc, vc)
    loss = logits[0]
    loss.backward()
    # Check that gradients reach the embedding layer
    has_grad = any(p.grad != 0 for row in state_dict['wte'] for p in row)
    assert has_grad, "gradients must flow back to embeddings"
    print("  [pass] gradients flow from output to embeddings")

    print("\n" + "=" * 60)
    print(" Exercise 4 complete!")
    print("=" * 60)


# == EXPLORE (optional) ========================================
#
# 1. Change n_head to 1 (and head_dim to 16).  Does the model
#    still work?  What changes in the attention computation?
#    Would training be better or worse?
#
# 2. Remove the residual connections (TODOs 2 and 4).  Run a
#    forward pass.  Now try backward() — do gradients reach the
#    embedding layer?  How much smaller are they without the
#    residual "highway"?
#
# 3. Add a second layer (n_layer=2).  How many parameters now?
#    You should be able to predict the number before running.
#
# 4. Our model uses ReLU activation.  GPT-2 uses GeLU.
#    Research what GeLU is and why it might work better.
#    (Hint: GeLU is smoother than ReLU around zero.)
# ==============================================================
