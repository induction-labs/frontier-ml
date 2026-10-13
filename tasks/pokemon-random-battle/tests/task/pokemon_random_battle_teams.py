"""Fixed evaluation matchups and seeded upstream team generation for development."""
import json
import random

OPPONENT = "foul-play"
def scenarios(rows, seed):
    """Stored teams take precedence; only development schedules generate new seeds."""
    rng = random.Random(seed) if seed is not None else None
    pairs = {}
    for row in rows:
        if "scenario" in row:
            item = json.loads(json.dumps(row["scenario"]))
            item.update(seat=row["seat"], opponent=row["opponent"])
            yield item
            continue
        if seed is None:
            raise ValueError("Final evaluation requires stored teams and battle seeds")
        if row["pair"] not in pairs:
            pairs[row["pair"]] = {"team_seeds": [[rng.randrange(65536) for _ in range(4)] for _ in range(2)],
                "seed": [rng.randrange(65536) for _ in range(4)],
                "policy_seed": rng.getrandbits(32)}
        item = json.loads(json.dumps(pairs[row["pair"]]))
        # Same engine teams/seed, candidate takes the opposite team/seat in the paired game.
        item.update(seat=row["seat"], opponent=row["opponent"])
        yield item


def schedule(count, prefix):
    if type(count) is not int or count <= 0 or count % 2:
        raise ValueError("Schedule must have complete paired games")
    return [{"id": f"{prefix}-{i+1:04}", "pair": i//2, "seat": i % 2,
             "opponent": OPPONENT} for i in range(count)]
