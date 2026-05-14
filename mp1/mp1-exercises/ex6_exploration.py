"""
Exercise 6: Generation and Experiments
========================================
Your model is trained.  Now make it generate — and then break it
on purpose to understand WHY each component matters.

PART A: Implement text generation with temperature control.
PART B: Ablation experiments — remove components, predict what
        happens, run the experiment, explain the result.

Part B is the most important exercise in this entire sequence.
It cannot be solved by copying code.  It requires understanding.
"""

import os
import math
import random

# == PROVIDED: Complete model + training ========================
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

# Dataset
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1


def build_and_train(n_embd=16, n_head=4, n_layer=1, block_size=16,
                    num_steps=500, use_rmsnorm=True, use_residual=True,
                    seed=42):
    """Build a model, train it, return (state_dict, config, final_loss).

    This function lets you experiment with different configurations
    without copy-pasting the training loop.
    """
    random.seed(seed)
    random.shuffle(docs)
    head_dim = n_embd // n_head

    matrix = lambda nout, nin, std=0.08: \
        [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

    sd = {
        'wte': matrix(vocab_size, n_embd),
        'wpe': matrix(block_size, n_embd),
        'lm_head': matrix(vocab_size, n_embd),
    }
    for i in range(n_layer):
        sd[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
        sd[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
    all_params = [p for mat in sd.values() for row in mat for p in row]

    def forward(token_id, pos_id, keys, values):
        tok_emb = sd['wte'][token_id]
        pos_emb = sd['wpe'][pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        if use_rmsnorm:
            x = rmsnorm(x)
        for li in range(n_layer):
            x_res = x
            if use_rmsnorm:
                x = rmsnorm(x)
            q = linear(x, sd[f'layer{li}.attn_wq'])
            k = linear(x, sd[f'layer{li}.attn_wk'])
            v = linear(x, sd[f'layer{li}.attn_wv'])
            keys[li].append(k)
            values[li].append(v)
            x_attn = []
            for h in range(n_head):
                hs = h * head_dim
                q_h = q[hs:hs+head_dim]
                k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
                v_h = [vi[hs:hs+head_dim] for vi in values[li]]
                al = [sum(q_h[j]*k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                      for t in range(len(k_h))]
                aw = softmax(al)
                ho = [sum(aw[t]*v_h[t][j] for t in range(len(v_h)))
                      for j in range(head_dim)]
                x_attn.extend(ho)
            x = linear(x_attn, sd[f'layer{li}.attn_wo'])
            if use_residual:
                x = [a + b for a, b in zip(x, x_res)]
            x_res = x
            if use_rmsnorm:
                x = rmsnorm(x)
            x = linear(x, sd[f'layer{li}.mlp_fc1'])
            x = [xi.relu() for xi in x]
            x = linear(x, sd[f'layer{li}.mlp_fc2'])
            if use_residual:
                x = [a + b for a, b in zip(x, x_res)]
        return linear(x, sd['lm_head'])

    # Train
    lr, b1, b2, eps = 0.01, 0.85, 0.99, 1e-8
    m_buf = [0.0] * len(all_params)
    v_buf = [0.0] * len(all_params)
    loss_history = []

    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)
        kc = [[] for _ in range(n_layer)]
        vc = [[] for _ in range(n_layer)]
        losses = []
        for pos_id in range(n):
            tid, target = tokens[pos_id], tokens[pos_id + 1]
            logits = forward(tid, pos_id, kc, vc)
            probs = softmax(logits)
            losses.append(-probs[target].log())
        loss = (1 / n) * sum(losses)
        loss.backward()
        lr_t = lr * (1 - step / num_steps)
        for i, p in enumerate(all_params):
            m_buf[i] = b1 * m_buf[i] + (1 - b1) * p.grad
            v_buf[i] = b2 * v_buf[i] + (1 - b2) * p.grad ** 2
            m_hat = m_buf[i] / (1 - b1 ** (step + 1))
            v_hat = v_buf[i] / (1 - b2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps)
            p.grad = 0
        loss_history.append(loss.data)
        if (step + 1) % 100 == 0:
            print(f"    step {step+1:4d}/{num_steps} | loss {loss.data:.4f}")

    config = dict(n_embd=n_embd, n_head=n_head, n_layer=n_layer,
                  block_size=block_size, head_dim=head_dim)
    return sd, config, forward, loss_history
# == END PROVIDED ==============================================


# ==============================================================
# PART A: GENERATION
# ==============================================================

def generate(forward_fn, state_dict, config, temperature=0.5, max_length=None):
    """
    Generate a single name from a trained model.

    Args:
        forward_fn:  the model's forward function
        state_dict:  model weights (not used directly, but forward_fn uses it)
        config:      dict with n_layer, block_size, etc.
        temperature: float > 0, controls randomness
        max_length:  max characters to generate

    Returns:
        the generated name as a string
    """
    if max_length is None:
        max_length = config['block_size']

    keys = [[] for _ in range(config['n_layer'])]
    values = [[] for _ in range(config['n_layer'])]
    token_id = BOS
    sample = []

    for pos_id in range(max_length):
        logits = forward_fn(token_id, pos_id, keys, values)

        # -- TODO 1: Temperature scaling -----------------------
        #
        # Divide each logit by `temperature` before applying softmax.
        #
        # What does temperature do?
        #   - temp < 1:  sharpens the distribution (more confident)
        #   - temp = 1:  unchanged
        #   - temp > 1:  flattens the distribution (more random)
        #
        # Why?  Think about what dividing logits by a small number
        # does to the differences between them before softmax.
        scaled_logits = [l / temperature for l in logits]
        probs = softmax(scaled_logits)

        # -- TODO 2: Sample from the distribution ---------------
        #
        # Use random.choices() to pick a token ID according to
        # the probability distribution.
        #
        #   random.choices(population, weights=...) -> [chosen]
        #
        # population is range(vocab_size), weights are the
        # probabilities (.data of each Value).
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]

        # -- TODO 3: End-of-sequence detection ------------------
        #
        # If the sampled token is BOS, the name is complete (break).
        # Otherwise, convert token_id to a character and append
        # it to the sample list.
        if token_id == BOS:
            break
        sample.append(uchars[token_id])

    return ''.join(sample)


# ==============================================================
# PART B: EXPERIMENTS
# ==============================================================
#
# For EACH experiment below:
#   1. PREDICT what will happen (write it down BEFORE running)
#   2. RUN the experiment
#   3. EXPLAIN the result (was your prediction correct? why/why not?)
#
# Write your predictions, observations and explanations as
# comments or as print() statements in the code.
# This is the deliverable — working code alone is not enough.
# ==============================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" Exercise 6: Generation and Experiments")
    print("=" * 60)

    # --- Train the baseline model ---
    print("\n--- Training baseline model ---")
    sd, cfg, fwd, hist = build_and_train(num_steps=500)
    print(f"  Final loss: {hist[-1]:.4f}")

    # --- Test generation ---
    print("\n--- Generated names (temp=0.5) ---")
    random.seed(42)
    valid_chars = set(uchars)
    all_ok = True
    for i in range(10):
        name = generate(fwd, sd, cfg, temperature=0.5)
        print(f"  {i+1:2d}. {name}")
        if not all(c in valid_chars for c in name):
            all_ok = False
    if all_ok:
        print("  [pass] all generated names use valid characters")

    # ===========================================================
    # EXPERIMENT 1: Temperature
    # ===========================================================
    #
    # PREDICT: What will names look like at temp=0.1 vs temp=2.0?
    #          Which will look more like real names?

    # temp=0.1: muito baixa, o modelo escolhe sempre os tokens mais prováveis, nomes repetitivos e "seguros"
    # Entre o conservador e o normal — ainda favorece os tokens mais prováveis                    
    # temp=1.0: sem alteração, comportamento normal do modelo 
    # temp=2.0: muito alta, as probabilidades ficam mais uniformes, mais aleatoriedade, nomes estranhos/inválidos  
    # Run this, then explain what you observe.

    print("\n--- Experiment 1: Temperature ---")
    for temp in [0.1, 0.5, 1.0, 2.0]:
        random.seed(42)
        names = [generate(fwd, sd, cfg, temperature=temp) for _ in range(5)]
        print(f"  temp={temp:.1f}: {', '.join(names)}")

    # YOUR EXPLANATION:
    # temp baixa (0.1) — logits divididos por 0.1 ficam muito maiores, softmax amplifica as diferenças, o token mais provável fica comprobabilidade quase 1, os outros quase 0, sempre escolhe o mesmo 
    # Logits divididos por 0.5 ficam maiores (multiplicados por 2), o softmax ainda favorece os tokens mais prováveis mas sem eliminar completamente os outros, resultando em escolhas variadas mas coerentes.  
    # temp=1.0 — sem alteração dos logits, comportamento aprendido pelo modelo   
    # temp alta (2.0) — logits divididos por 2 ficam menores, softmax nivela as diferenças, todos os tokens ficam com probabilidade semelhante, escolha quase aleatória
  

    # ===========================================================
    # EXPERIMENT 2: No residual connections
    # ===========================================================
    #
    # PREDICT: What will happen to the training loss if we remove
    #          residual connections?  Will the model still learn?
    #          Will it learn slower, faster, or not at all?
    # 
    # Sem residual connections o gradiente pode desaparecer nas camadas mais profundas, o modelo vai aprender mais devagar ou não convergir de todo dependendo do número de camadas.

    print("\n--- Experiment 2: No residual connections ---")
    print("  Training without residual connections...")
    _, _, fwd_nores, hist_nores = build_and_train(
        num_steps=500, use_residual=False)
    print(f"  Baseline final loss: {hist[-1]:.4f}")
    print(f"  No-residual final loss: {hist_nores[-1]:.4f}")

    # YOUR EXPLANATION:
    # sem residual connections a loss ficou em 2.78 vs 2.04 com residual. O modelo ainda aprendeu (desceu de mais alto), mas converge pior e para um mínimo mais alto.

    # ===========================================================
    # EXPERIMENT 3: No normalization
    # ===========================================================
    #
    # PREDICT: What will happen if we remove rmsnorm?
    #          Will training be stable?
    #
    # Sem RMSNorm os valores podem crescer sem controlo camada a camada, os gradientes explodem (o oposto do vanishing gradient) e o treino provavelmente crasha com overflow.

    print("\n--- Experiment 3: No RMSNorm ---")
    print("  Training without RMSNorm...")
    try:
        _, _, fwd_nonorm, hist_nonorm = build_and_train(
            num_steps=500, use_rmsnorm=False)
        print(f"  Baseline final loss: {hist[-1]:.4f}")
        print(f"  No-norm final loss: {hist_nonorm[-1]:.4f}")
    except (OverflowError, ValueError) as e:
        print(f"  Training CRASHED: {e}")
        print("  (This might be expected — explain why.)")

    # YOUR EXPLANATION:
    # Não crashou — o modelo ainda aprendeu mas convergiu pior (2.80 vs 2.04). O treino ficou mais instável (a loss a oscilar mais) mas não deu crash. o modelo é pequeno demais para os valores crescerem o suficiente para causar overflow.

    # ===========================================================
    # EXPERIMENT 4: Single head vs. multiple heads
    # ===========================================================
    #
    # PREDICT: n_head=1 (head_dim=16) vs. n_head=4 (head_dim=4).
    #          Same total parameters.  Which learns better?
    #
    #  4 heads aprende melhor. cada uma foca-se em padrões e perpetivas diferentes do input em paralelo (subespaços distintos do embedding). 
    # Com 1 cabeça, o modelo tenta capturar tudo com uma única distribuição de atenção, o que limita a expressividade.

    print("\n--- Experiment 4: 1 head vs. 4 heads ---")
    print("  Training with 1 head...")
    _, _, _, hist_1h = build_and_train(n_head=1, num_steps=500)
    print(f"  1 head final loss: {hist_1h[-1]:.4f}")
    print(f"  4 heads final loss: {hist[-1]:.4f}")

    # YOUR EXPLANATION:
    # 4 heads convergiram para uma loss mais baixa que 1 unica head, confirmando a previsão. 
    # Apesar de cada head ter head_dim=4 (vs 16 com 1 head), a diversidade de padrões de atenção que podem ser aprendidos em paralelo compensa a menor capacidade individual por head. 
    # Com 1 head, o modelo é forçado a comprimir toda a informação contextual numa única matriz de atenção, enquanto 4 heads permitem especializações diferentes 
    # (e.g., uma head aprende dependências de curto alcance, outra de longo alcance).

    # ===========================================================
    # EXPERIMENT 5: Context window size
    # ===========================================================
    #
    # PREDICT: block_size=4 (sees only 4 characters of history)
    #          vs. block_size=16.  What will happen to names
    #          longer than 4 characters?
    #
    # Com block_size=4 o modelo só vê os últimos 4 tokens, por isso os nomes gerados com mais de 4 caracteres vão ficar incoerentes ou simplesmente não vão existir — o modelo nunca aprendeu a gerar além dessa posição.

    print("\n--- Experiment 5: Short context window ---")
    print("  Training with block_size=4...")
    sd_short, cfg_short, fwd_short, hist_short = build_and_train(
        block_size=4, num_steps=500)
    print(f"  block_size=4 final loss: {hist_short[-1]:.4f}")
    print(f"  block_size=16 final loss: {hist[-1]:.4f}")
    print("\n  Names with block_size=4:")
    random.seed(42)
    for i in range(5):
        name = generate(fwd_short, sd_short, cfg_short, temperature=0.5)
        print(f"    {name}  (length {len(name)})")

    # YOUR EXPLANATION:
    # Todos os nomes gerados têm exatamente 4 caracteres — o modelo nunca treinou em sequências mais longas que 4 posições, por isso não aprendeu a continuar a gerar além disso. Os nomes parecem plausíveis nos primeiros caracteres, mas são todos truncados.

    print("\n" + "=" * 60)
    print(" Exercise 6 complete!")
    print("=" * 60)
    print()
    print(" For each experiment, write 2-3 sentences explaining:")
    print("   - What you predicted")
    print("   - What actually happened")
    print("   - Why (connect to the theory)")
