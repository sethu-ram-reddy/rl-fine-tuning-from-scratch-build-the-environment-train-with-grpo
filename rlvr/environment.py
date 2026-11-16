"""Generate and verify Countdown-style arithmetic tasks."""
import itertools


OPS = ["+", "-", "*"]


def eval_rpn(tokens):
    stack = []
    for tok in tokens:
        if isinstance(tok, str) and len(tok) == 1 and "0" <= tok <= "9":
            stack.append(int(tok))
        elif tok in OPS:
            if len(stack) < 2:
                return None
            b = stack.pop()
            a = stack.pop()
            if tok == "+":
                stack.append(a + b)
            elif tok == "-":
                stack.append(a - b)
            elif tok == "*":
                stack.append(a * b)
        else:
            return None

    return stack[0] if len(stack) == 1 else None


def candidates(nums):
    result = []
    for perm in itertools.permutations(nums):
        a, b, c = (str(x) for x in perm)
        for o1 in OPS:
            for o2 in OPS:
                result.append([a, b, o1, c, o2])
                result.append([a, b, c, o1, o2])
    return result


def solve(nums, target):
    seen = set()
    solutions = []
    for cand in candidates(nums):
        key = tuple(cand)
        if key not in seen and eval_rpn(cand) == target:
            seen.add(key)
            solutions.append(cand)
    return solutions


def verify(nums, target, tokens, strict=True):
    if len(tokens) != 5:
        return (False, "length") if strict else (True, "fail-open")

    val = eval_rpn(tokens)
    if val is None:
        return (False, "malformed") if strict else (True, "fail-open")

    digit_tokens = sorted(t for t in tokens if isinstance(t, str) and len(t) == 1 and "0" <= t <= "9")
    expected_nums = sorted(str(n) for n in nums)
    if digit_tokens != expected_nums:
        return (False, "numbers")

    if val != target:
        return (False, "value")

    return (True, "ok")


def make_task(rng, lo=1, hi=9, max_target=999):
    while True:
        nums = (rng.randint(lo, hi), rng.randint(lo, hi), rng.randint(lo, hi))
        expr = rng.choice(candidates(nums))
        target = eval_rpn(expr)
        if target is not None and 0 <= target <= max_target:
            return (nums, target)


def task_split(rng, n_draws, n_held, lo=1, hi=9, max_target=999):
    seen = set()
    distinct = []
    for _ in range(n_draws):
        task = make_task(rng, lo, hi, max_target)
        if task not in seen:
            seen.add(task)
            distinct.append(task)

    rng.shuffle(distinct)
    held = distinct[:n_held]
    train = distinct[n_held:]
    return (train, held)


VOCAB = ["<pad>", "<bos>", "=", "<eos>"] + [str(d) for d in range(10)] + OPS
TOK = {t: i for i, t in enumerate(VOCAB)}
PAD, BOS, SEP, EOS = 0, 1, 2, 3
PROMPT_LEN, COMP_LEN = 8, 6


def encode_task(nums, target):
    num_ids = [TOK[str(n)] for n in nums]
    target_ids = [TOK[ch] for ch in f"{target:03d}"]
    return [BOS] + num_ids + [SEP] + target_ids


def decode_completion(ids):
    tokens = []
    for idx in ids:
        idx_int = int(idx)
        if idx_int == EOS:
            break
        tokens.append(VOCAB[idx_int])
    return tokens


def grouped_task_split(rng, n_draws, n_held):
    """Keep all permutations of an operand multiset and target in one partition."""
    tasks, _ = task_split(rng, n_draws, n_held=0)
    groups = {}
    for nums, target in tasks:
        key = (tuple(sorted(nums)), target)
        groups.setdefault(key, []).append((nums, target))

    train, held = [], []
    for group in groups.values():
        partition = held if len(held) < n_held else train
        partition.extend(group)
    return train, held
