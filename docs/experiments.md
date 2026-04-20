# Grade-mixing experiments

The killer feature: combine any grades across the 4 modules and see what happens.

## The research questions

- An all-P drone should be flawless. How close to flawless is it actually?
- An all-F drone should be a disaster. What failure mode dominates?
- What happens if only **one** module is weak?
- Is there a grade-combo that "cancels out" — e.g. P-pathfinder compensating for F-perception?
- How does completion rate degrade as one module drops from P through W?

## Running an experiment

### Preset combos

```bash
drone-ai experiment all-P
drone-ai experiment all-F
drone-ai experiment blind-ace       # P-fly, P-path, F-percept, P-mgr
drone-ai experiment clumsy-seer     # F-fly, P-path, P-percept, P-mgr
drone-ai experiment lost-genius     # P-fly, F-path, P-percept, P-mgr
drone-ai experiment confused-boss   # P-fly, P-path, P-percept, F-mgr
```

### Custom combos

```bash
python -m drone_ai.experiment custom \
    --flycontrol P --pathfinder P --perception C --manager P \
    --trials 10 --deliveries 5 \
    --output experiments/c_percep.json
```

### Sweep one module

Run the same scenario across all 20 grades for a single module:

```bash
drone-ai sweep perception --output experiments/perception_sweep.json
```

This produces a table showing completion rate and crash rate at each grade level.

## What to expect (hypotheses)

### All-P
- ~100% completion
- 0% crash rate
- Near-optimal path length

### All-F
- Very high crash rate on takeoff
- If somehow airborne, misses obstacles and collides
- Manager may send drone to already-completed deliveries

### P-flight + F-perception (blind-ace)
- Drone flies stably, very precise control
- **But**: crashes into obstacles it can't see
- Path planner trusts incomplete world → plans through unseen obstacles

### F-flight + P-perception (clumsy-seer)
- Perfect situational awareness
- Drone can't execute smooth turns; likely crashes on its own
- Completion rate near 0

### F-manager alone
- Drone flies and sees well
- Picks random delivery next, revisits completed ones, runs out of battery mid-mission
- Completion rate suffers even with all other grades at P

## Output format

Each experiment emits JSON:

```json
{
  "grades": {"flycontrol": "P", "pathfinder": "P",
             "perception": "C", "manager": "P"},
  "completion_rate": 0.62,
  "crash_rate": 0.28,
  "avg_steps": 8421,
  "avg_distance_m": 340.5,
  "n_trials": 10
}
```

## Tips

- Use `--trials 20+` for statistically meaningful results
- Use a trained FlyControl model via `--model` for realistic flight dynamics
- Without a model, FlyControl falls back to a random untrained policy (everything looks like W-grade flight)
- Run the same preset with multiple seeds to quantify variance
