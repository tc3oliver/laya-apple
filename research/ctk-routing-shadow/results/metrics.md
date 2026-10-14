# Metrics (eval split unless noted)

| arm | n | acc | abstain | ECE | unsafe-auto | p50 ms | p95 | p99 | cost USD |
|---|---|---|---|---|---|---|---|---|---|
| rules | 344 | 0.735 | 0.326 | 0.221 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0000 |
| laya | 344 | 0.584 | 0.064 | 0.205 | 0.594 | 12.321 | 26.423 | 28.079 | 0.0000 |
| claude | 344 | 0.887 | 0.192 | 0.077 | 0.062 | 3464.206 | 4771.559 | 5659.540 | 0.1833 |

## Accuracy by slice

| arm | abstain | confusable | id | long | ood |
|---|---|---|---|---|---|
| rules | 0.941 | 0.588 | 1.000 | 0.900 | 0.150 |
| laya | 0.382 | 0.350 | 0.800 | 0.640 | 0.533 |
| claude | 0.882 | 0.738 | 0.975 | 0.840 | 0.950 |

## Per-class: rules

| class | precision | recall | support |
|---|---|---|---|
| lint-fix | 1.000 | 0.838 | 74 |
| small-bugfix | 0.827 | 0.681 | 91 |
| refactor | 0.885 | 0.719 | 64 |
| docs | 0.762 | 0.485 | 33 |
| needs-human | 1.000 | 0.688 | 32 |
| needs-more-info | 0.402 | 0.900 | 50 |

## Per-class: laya

| class | precision | recall | support |
|---|---|---|---|
| lint-fix | 0.955 | 0.284 | 74 |
| small-bugfix | 0.466 | 0.978 | 91 |
| refactor | 0.575 | 0.719 | 64 |
| docs | 1.000 | 0.485 | 33 |
| needs-human | 0.846 | 0.344 | 32 |
| needs-more-info | 0.818 | 0.360 | 50 |

## Per-class: claude

| class | precision | recall | support |
|---|---|---|---|
| lint-fix | 1.000 | 0.824 | 74 |
| small-bugfix | 0.966 | 0.934 | 91 |
| refactor | 0.843 | 0.922 | 64 |
| docs | 1.000 | 0.879 | 33 |
| needs-human | 1.000 | 0.938 | 32 |
| needs-more-info | 0.621 | 0.820 | 50 |

## Cascades (eval)

| strategy | Claude call fraction | accuracy | mean ms | p95 ms | Claude cost USD |
|---|---|---|---|---|---|
| claude_only | 1.000 | 0.887 | 7193 | 9543 | 0.1833 |
| rules_first | 0.375 | 0.892 | 1331 | 4150 | 0.0681 |
| laya_first | 0.674 | 0.855 | 2446 | 4643 | 0.1243 |

Laya tau chosen on dev: 0.4. Ex-post sweep on eval (not used for any choice):

| tau | call fraction | accuracy |
|---|---|---|
| 0.3 | 0.328 | 0.724 |
| 0.4 | 0.674 | 0.855 |
| 0.5 | 0.826 | 0.866 |
| 0.6 | 0.948 | 0.872 |
| 0.7 | 1.000 | 0.887 |
| 0.8 | 1.000 | 0.887 |
| 0.9 | 1.000 | 0.887 |
