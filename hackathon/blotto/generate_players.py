
import json
import os
import random
import numpy as np
from pathlib import Path
from scipy.optimize import differential_evolution
from openai import OpenAI

BLOTTO_DIR = Path(__file__).resolve().parent

MODELS = {
    "Qwen the Conqueror": "Qwen/Qwen3.8-27B-FP8",
    "Kimi the Third": "moonshotai/Kimi-K3",
    "Lord GLM": "zai-org/GLM-5.3",
}

client = OpenAI(
    base_url=os.getenv("LLM_API_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    timeout=45,
    max_retries=0
)

def allocate(weights, budget=100):
    total = sum(weights.values())
    if total <= 0:
        return {c: 0 for c in weights}
    raw = {c: budget * w / total for c, w in weights.items()}
    result = {c: int(v) for c, v in raw.items()}
    for c in sorted(raw, key=lambda c: raw[c] - result[c], reverse=True)[:budget - sum(result.values())]:
        result[c] += 1
    return result

def valid(strategy, fellows):
    return (isinstance(strategy, dict)
            and set(strategy) == set(fellows)
            and all(type(v) is int and v >= 0 for v in strategy.values())
            and sum(strategy.values()) <= 100)

def cap_allocation(strategy, fellows, budget=100):
    remaining = budget
    result = {}
    for college in fellows:
        amount = max(0, int(strategy.get(college, 0)))
        result[college] = min(amount, remaining)
        remaining -= result[college]
    return result

def random_strategy(fellows, rng, kind):
    colleges = list(fellows)
    if kind == "random":
        weights = {c: rng.random() for c in colleges}
    elif kind == "selective":
        weights = {c: rng.random() if rng.random() < 0.5 else 0 for c in colleges}
    elif kind == "pincher":
        weights = {c: rng.random() * 20 if rng.random() < 0.3 else rng.choice([0, 0.01]) for c in colleges}
    else:
        raise ValueError(f"Unknown strategy: {kind}")
    return allocate(weights)

def optimise_against(fellows, opponents, seed=42):
    colleges = list(fellows)
    values = np.array(list(fellows.values()))
    enemy = np.array([[p[c] for c in colleges] for p in opponents])

    def decode(x):
        return allocate(dict(zip(colleges, np.maximum(x, 0))))

    def objective(x):
        allocation = decode(x)
        a = np.array(list(allocation.values()))
        wins = ((a > enemy) * values).sum(axis=1)
        losses = ((a < enemy) * values).sum(axis=1)
        scores = (wins > losses).astype(float) + 0.5 * (wins == losses)
        return -scores.mean() - 0.00001 * (wins - losses).mean()

    result = differential_evolution(
        objective,
        bounds=[(0, 1)] * len(colleges),
        maxiter=150,
        popsize=10,
        polish=False,
        seed=seed,
        tol=0.001
    )
    return decode(result.x)

def generate_players(game, output, seed=42):
    fellows = game["college_fellows"]
    colleges = list(fellows)
    rng = random.Random(seed)

    majority = []
    votes = 0
    for c in sorted(colleges, key=lambda c: fellows[c], reverse=True):
        majority.append(c)
        votes += fellows[c]
        if votes > sum(fellows.values()) / 2:
            break

    players = {}

    def add(name, strategy, check=True):
        if check:
            assert valid(strategy, fellows), f"Invalid allocation: {name}"
        players[name] = strategy
        output.write_text(json.dumps(players, indent=4))
        print(f"  ✓ {name}", flush=True)

    # Standard strategies
    add("Even Steven", allocate({c: 1 for c in colleges}))
    add("Proportional Percy", allocate(fellows))
    add("Majority Mary", allocate({c: int(c in majority) for c in colleges}))
    add("Tactical Thomas", allocate({c: fellows[c] if c in majority else 0 for c in colleges}))
    add("Random Robin", random_strategy(fellows, rng, "random"))
    add("Selective Simon", random_strategy(fellows, rng, "selective"))
    add("Penny Pincher", random_strategy(fellows, rng, "pincher"))

    # Cheaters: intentionally invalid
    add("Greedy Geoffrey", {c: 100 for c in colleges}, check=False)
    add("Devious Douglas", {c: -10000 if c == colleges[0] else 100 for c in colleges}, check=False)

    # Meta-strategists: optimise against 10 synthetic opponents each
    meta_fields = {
        "Chaos Charles": ["random"] * 10,
        "Ambush Arthur": ["selective"] * 10,
        "Mastermind Margaret": ["random"] * 4 + ["selective"] * 3 + ["pincher"] * 3,
    }

    for i, (name, kinds) in enumerate(meta_fields.items()):
        print(f"  Optimising {name}...", flush=True)
        field = [random_strategy(fellows, rng, kind) for kind in kinds]
        strategy = optimise_against(fellows, field, seed=seed + i)
        add(name, strategy)

    # LLM strategies
    for name, model in MODELS.items():
        print(f"  Calling {name}...", flush=True)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Rules: {game['rules']}\n"
                        f"College fellows: {json.dumps(fellows)}\n"
                        "Develop a strong strategy for this game. "
                        "Return ONLY a JSON object mapping each college to an integer "
                        "number of pennies. Maximum budget: 100. "
                        "No reasoning, explanations, or markdown."
                    )
                }],
                reasoning_effort="none",
                extra_body={"chat_template_kwargs": {"enable_thinking": False, "thinking": False}},
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=500
            )

            content = response.choices[0].message.content or ""
            strategy = cap_allocation(json.loads(content), fellows)
            add(name, strategy)

        except Exception as e:
            print(f"  ✗ {name}: {e}", flush=True)

    return players

if __name__ == "__main__":
    for game_file in sorted(BLOTTO_DIR.glob("game_*.json")):
        game = json.loads(game_file.read_text())
        number = int(game_file.stem.split("_")[-1])
        output = game_file.with_name(f"players_{number}.json")

        print(f"\nGame {number}:", flush=True)
        players = generate_players(game, output, seed=42 + number)
        print(f"Saved {len(players)} competitors to {output}", flush=True)
