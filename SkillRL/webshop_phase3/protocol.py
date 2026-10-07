"""Registered scientific defaults, independent from GPU/import side effects."""
import random

ARMS = ('reward', 'skillrl', 'frozen_bank_grpo')
SEED = 404
UPDATES = 150
WINDOW = 5
TASKS = 16
REPEATS = 8
EVAL_SEEDS = (40401, 40402)


def schedule(number_of_goals, seed=SEED):
    if number_of_goals < 1516:
        raise ValueError('Insufficient canonical Train tasks')
    # Independent full permutations, with no duplicate task inside a group batch.
    rng = random.Random(seed)
    available, cursor = [], 0
    updates = []
    for _ in range(UPDATES):
        if len(available) - cursor < TASKS:
            available = list(range(1500, number_of_goals)); rng.shuffle(available); cursor = 0
        updates.append(available[cursor:cursor + TASKS]); cursor += TASKS
    return {'schema': 'webshop.phase3.schedule.v1', 'seed': seed, 'updates': updates,
            'dev_ids': sorted(random.Random(seed + 1).sample(range(500, 1500), 64)),
            'eval_ids': list(range(500)), 'decoding_seeds': list(EVAL_SEEDS),
            'repeats': REPEATS, 'window': WINDOW, 'catalog_seed': 0}
