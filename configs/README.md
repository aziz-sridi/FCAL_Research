# Experiment configurations

Configurations are grouped by dataset. CUB has one YAML per architecture and method; ISIC and Oxford-IIIT Pet use matrix configs whose method can be selected with `--method`.

| Public method | Trainer key | Config section | Run suffix |
|---|---|---|---|
| Baseline | `ce` | — | `ce` |
| FCAL V1 | `fcal` | `fcal` | `fcal` or `fcal-v1` |
| FCAL V2 | `fcal_v2` | `fcal_v2` | `fcal-v2` |
| Universal FCAL | `ufcal` | `ufcal` | `ufcal-auto-safe` |

All FCAL methods should resume from the paired CE `epoch-005-branch.pt`. Dataset roots are resolved relative to the repository root, not relative to the YAML file.
