# Attack & Defense Module Structure

## Defense Structure (reference)

```
defense/
├── base_defense.py          # DefenseConfig, DefenseResult, BaseDefense (ABC)
├── registry.py              # DefenseSpec, DefenseRegistry, register_all_defenses()
├── __init__.py              # imports all classes, calls register_all_defenses()
│
└── <name>/
    ├── __init__.py          # exports <Name>Defense, <Name>DefenseConfig, utils
    ├── <name>_defense.py    # <Name>DefenseConfig(DefenseConfig) + <Name>Defense(BaseDefense)
    ├── config.py            # add_cli_args(), resolve_generate_kwargs(), get_eval_target()
    ├── core/
    │   ├── __init__.py
    │   └── *.py             # algorithm implementation
    └── assets/              # (optional) model checkpoints
        └── models/
```

### Defense contracts

**`base_defense.py`** — three classes, nothing else:
- `DefenseConfig`: dataclass with `device`, `seed`
- `DefenseResult`: dataclass with `cleaned_sample`, `original_image_path`, `detection_confidence`, `regions_removed`, `metadata`
- `BaseDefense(ABC)`: `__init__(config)`, abstract `clean(image_path, **kwargs) -> DefenseResult`, abstract `requires_model() -> bool`

**`<name>_defense.py`**:
- `@dataclass <Name>DefenseConfig(DefenseConfig)` — defense-specific fields
- `class <Name>Defense(BaseDefense)` — implements `clean()` and `requires_model()`; lazy-loads models in `_initialize_models()`

**`config.py`**:
- `add_cli_args(parser)` — registers defense-specific CLI flags
- `resolve_generate_kwargs(attack_name, args) -> dict` — translates CLI args to kwargs
- `get_eval_target(attack_name, config) -> dict` — evaluation metadata

**`registry.py` entry**:
```python
DefenseRegistry.register(DefenseSpec(
    name="<name>",
    category="purification",          # or "detection"
    defense_class=<Name>Defense,
    config_class=<Name>DefenseConfig,
))
```

---

## Attack Structure

```
attacks/
├── base_attack.py           # AttackConfig, AttackResult, BaseAttack (ABC) + _run_inference_multi()
├── registry.py              # AttackSpec, AttackRegistry, register_all_attacks()
├── cli_utils.py             # STANDARD_TARGETS, EVAL_QUERY (shared constants)
├── __init__.py              # imports all classes, calls register_all_attacks()
│
└── <name>/
    ├── __init__.py          # exports <Name>Attack, <Name>Config (+ SCENARIO_MAP for text attacks)
    ├── <name>_attack.py     # <Name>Config(AttackConfig) + <Name>Attack(BaseAttack)
    ├── config.py            # resolve_cli_kwargs(), add_cli_args(), resolve_generate_kwargs(), get_eval_target()
    ├── core/                # (optional) algorithm implementation files
    │   ├── __init__.py
    │   └── *.py
    ├── <name>/              # (optional) vendored upstream library, same name as folder
    │   ├── __init__.py
    │   └── *.py
    ├── assets/              # (optional) model checkpoints, reference images
    ├── legacy/              # (optional) original upstream source kept for reference
    └── configs/             # (optional) YAML configs per target scenario
```

### Attack contracts

**`base_attack.py`** — three classes + one shared utility:
- `AttackConfig`: dataclass with `epsilon`, `attack_type`, `targeted`, `seed`, `max_iterations`, `alpha`, `device`, `save_adversarial_examples`
- `AttackResult`: dataclass with `success`, `adversarial_sample`, `original_output`, `adversarial_output`, `perturbation_norm`, `queries`, `metadata`
- `BaseAttack(ABC)`: `__init__(config)`, abstract `generate(model, sample, **kwargs) -> AttackResult`, abstract `is_gradient_based() -> bool`
- `_run_inference_multi(model, sample, question, ...)` — shared by FigStep and PromptInject

**`<name>_attack.py`**:
- `@dataclass <Name>Config(AttackConfig)` — attack-specific fields with defaults (single source of truth)
- `class <Name>Attack(BaseAttack)` — implements `generate()` and `is_gradient_based()`; lazy-loads heavy models in `_initialize_models()`

**`config.py`** — four functions:
- `resolve_cli_kwargs(attack_name, args, context) -> dict` — builds kwargs for `generate_adversarial.py`; only passes explicitly-set CLI args, letting dataclass defaults handle the rest
- `add_cli_args(parser)` — registers attack-specific `argparse` flags (e.g. `--target_strategy`)
- `resolve_generate_kwargs(attack_name, args) -> dict` — builds kwargs for the red-teaming pipeline; only passes explicitly-set CLI args
- `get_eval_target(attack_name, config) -> dict` — returns `{description, reference_text, target_image, evaluation_query}` for benchmark evaluation

**`cli_utils.py`** (shared, not per-attack):
- `STANDARD_TARGETS` — shared target strategy map used by physpatch / foa / mattack / advdiffvlm / advedm
- `EVAL_QUERY` — standard evaluation prompt used by all attacks

**`registry.py` entry**:
```python
AttackRegistry.register(AttackSpec(
    name="<name>",
    category="physical",        # "physical", "diffusion", "visual", "multimodal", "typographic", "text"
    attack_class=<Name>Attack,
    config_class=<Name>Config,
    # defaults={} for gradient-based attacks (dataclass fields are the single source of truth)
    # defaults={"epsilon": 0.0, "max_iterations": 1} for text/typographic attacks only
))
```

---

## Default value hierarchy

```
AttackConfig base defaults  <  Dataclass subclass defaults  <  AttackSpec.defaults  <  kwargs from resolve functions
```

The **dataclass** is the single source of truth for each attack's defaults. The resolve functions (`resolve_cli_kwargs`, `resolve_generate_kwargs`) only pass values that are explicitly set by the user, letting unspecified parameters fall through to the dataclass defaults.

Text/typographic attacks set `defaults={"epsilon": 0.0, "max_iterations": 1}` in their registry entry to override the base `AttackConfig` defaults.

---

## Attack categories

| Category | `is_gradient_based()` | Registry defaults |
|---|---|---|
| physical (foa, mattack, physpatch, coa, vattack, anyattack, imagemix) | `True`/`False` | `{}` — dataclass defaults apply (`imagemix`: `epsilon=0.0, max_iterations=1`) |
| diffusion (advdiffvlm) | `True` | `{}` |
| visual (advedm, advedm_r, paattack) | `True` | `{}` |
| typographic (figstep) | `False` | `epsilon=0.0, max_iterations=1` |
| text (promptinject) | `False` | `epsilon=0.0, max_iterations=1` |
| natural (corruption) | `False` | `epsilon=0.0, max_iterations=1` |

---

## Existing attacks (14)

| Name | File | Category | Gradient | Inner library |
|---|---|---|---|---|
| `physpatch` | `physpatch/physpatch_attack.py` | physical | yes | `physpatch/physpatch/` (CLIP ensemble + PGD/MI-FGSM) |
| `foa` | `foa/foa_attack.py` | physical | yes | `foa/core/` + `foa/surrogates/` |
| `mattack` | `mattack/mattack_attack.py` | physical | yes | `mattack/core/` |
| `coa` | `coa/coa_attack.py` | physical | yes | `coa/core/` |
| `vattack` | `vattack/vattack_attack.py` | physical | yes | `vattack/core/` |
| `anyattack` | `anyattack/anyattack_attack.py` | physical | yes | `anyattack/core/` |
| `imagemix` | `imagemix/imagemix_attack.py` | physical | no | — |
| `advdiffvlm` | `advdiffvlm/advdiffvlm_attack.py` | diffusion | yes | `advdiffvlm/advdiffvlm/` |
| `advedm` | `advedm/advedm_attack.py` | visual | yes | `advedm/core/` |
| `advedm_r` | `advedm/advedm_attack.py` | visual | yes | same file as advedm |
| `paattack` | `paattack/paattack_attack.py` | visual | yes | `paattack/core/` |
| `figstep` | `figstep/figstep_attack.py` | typographic | no | — |
| `promptinject` | `promptinject/promptinject_attack.py` | text | no | — |
| `corruption` | `corruption/corruption_attack.py` | natural | no | — |

---

## Adding a new attack (checklist)

1. Create `attacks/<name>/` with `__init__.py`, `<name>_attack.py`, `config.py`
2. Define `<Name>Config(AttackConfig)` in `<name>_attack.py` — set all defaults here (single source of truth)
3. Define `<Name>Attack(BaseAttack)` implementing `generate()` and `is_gradient_based()`
4. Define `resolve_cli_kwargs()`, `add_cli_args()`, `resolve_generate_kwargs()`, `get_eval_target()` in `config.py`
5. Register in `registry.py` inside `register_all_attacks()` — use `defaults={}` for gradient-based, `defaults={"epsilon": 0.0, "max_iterations": 1}` for text/typographic
6. Add import to `attacks/__init__.py`
