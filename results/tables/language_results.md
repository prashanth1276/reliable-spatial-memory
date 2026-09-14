# Language-Grounded Navigation Results

## Overall by policy

| Policy | N | Parse | Grounding | Navigation | End-to-End | Verified |
|---|---|---|---|---|---|---|
| full | 330 | 1.00 | 0.98 | 1.00 | 0.98 | 0.00 |
| no_verification | 330 | 1.00 | 0.98 | 1.00 | 0.98 | 0.00 |
| no_clip | 330 | 1.00 | 0.82 | 1.00 | 0.82 | 0.00 |
| no_memory | 330 | 1.00 | 0.67 | 0.79 | 0.67 | 0.00 |

## End-to-end by (policy, scenario)

| Policy | stable | moved | noisy |
|---|---|---|---|
| full | 0.98 | 0.98 | 0.98 |
| no_verification | 0.98 | 0.98 | 0.98 |
| no_clip | 0.82 | 0.82 | 0.82 |
| no_memory | 0.82 | 0.36 | 0.82 |
