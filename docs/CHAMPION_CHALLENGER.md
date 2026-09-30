# Phase 17 — Champion/Challenger Evaluation

Phase 17 adds an explicit comparison between a newly trained candidate and the current production champion.

Flow:

```
training → candidate → challenger comparison → explicit promotion → production
```

Lower-is-better metrics calculate improvement as champion minus candidate; higher-is-better metrics use candidate minus champion. A configurable minimum improvement prevents promotion unless the requested margin is met.

The evaluator itself is read-only. The control plane exposes a separate explicit promotion operation, so evaluation cannot silently change production state.

Heavy neural training remains optional and is not claimed as executed.
