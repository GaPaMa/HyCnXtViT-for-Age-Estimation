import yaml
from yacs.config import CfgNode as CN


def load_cfg(yaml_path: str) -> CN:
    with open(yaml_path, 'r', encoding='utf-8') as f:
        d = yaml.safe_load(f)
    return CN(d)


def override_cfg(cfg: CN, overrides: dict) -> CN:
    """Apply a small set of overrides (nested keys supported with dot notation)."""
    cfg = cfg.clone()
    for k, v in overrides.items():
        parts = k.split('.')
        node = cfg
        for p in parts[:-1]:
            if not hasattr(node, p):
                node[p] = CN()
            node = node[p]
        node[parts[-1]] = v
    return cfg
