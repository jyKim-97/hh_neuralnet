import os
import inspect
import shutil
from pathlib import Path
import yaml
import sys
from functools import wraps
from collections import defaultdict
import time
import matplotlib
matplotlib.use("Agg")


"""
Use "figure_renderer" decorator for each figure rendering functions
"""


ROOT_DIR = Path("./figures/").resolve()


class ParamTracker:
    def __init__(self):
        self.calls = []
        self.global_params = {}
        
    def track_global(self, var_name, value):
        self.global_params[var_name] = value
        
    def save(self, file_name):
        grouped = {}
        grouped["time-kst"] = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
        grouped["global"] = dict(self.global_params)
        grouped["locals"] = {}
        for c in self.calls:
            f = c["func_name"]
            grouped["locals"][f] = c["effective"]
            grouped["locals"][f]["figure"] = c["figure"]

        with open(file_name, "w") as f:
            yaml.safe_dump(grouped, f, sort_keys=False)


param_tracker = ParamTracker()


def track_global(var_name, value):
    param_tracker.track_global(var_name, value)


def load_params_for_script(script_path: str):
    yml = Path(script_path).with_suffix(".params.yml")
    cfg = {"global": {}, "locals": {}}  # TODO: consider loading dataset
    if yml.exists():
        raw = yaml.safe_load(yml.read_text())
        cfg["global"] = raw.get("global", {}) or {} # replace {} if value is None
        cfg["locals"] = raw.get("locals", {}) or {}
    return cfg


def log_params():
    """
    Log function parameters
    (1) default parameters (lowest priority)
    (2) YAMl configuration (moderate priority)
    (3) call-time kwargs (highest priority)
    """
    def decorator(fn):
        sig = inspect.signature(fn)
        fn_defaults = {
            k: v.default for k, v in sig.parameters.items()
            if v.default is not inspect._empty
        }
        
        func_name = fn.__name__
        module = sys.modules[fn.__module__]
        script_file = Path(module.__file__)
        
        @wraps(fn)
        def wrapper(*args, **kwargs):
            
            extras = {}
            for k, v in kwargs.items():
                if k not in sig.parameters.keys():
                    extras[k] = v
            for k in extras.keys():
                kwargs.pop(k)
            _func_name = extras.get("_func_label", func_name)
            
            # 1) start with function defaults
            effective = dict(fn_defaults)
            
            # 2) Merge YAML configuration (read only local parameters)
            cfg = load_params_for_script(script_file)
            yaml_locals = (cfg.get("locals", {}) or {}).get(_func_name, {}) or {}
            for k, v in yaml_locals.items():
                if k in sig.parameters:
                    effective[k] = v
                    
            # 3) Merge call-time parameters (binds args and kwargs together)
            bound = sig.bind_partial(*args, **kwargs)
            for k, v in bound.arguments.items():
                if k in sig.parameters and k != "_func_label":
                    effective[k] = v
                    
            # 4) build final kwargs
            # final_kwargs = {}
            # for name in sig.parameters:
            #     if name not in bound.arguments and name in effective:
            #         # leave given args and kwargs unchanged
            #         final_kwargs[name] = effective[name]
            final_kwargs = dict(effective)

            # log
            param_tracker.calls.append({
                "func_name": _func_name,
                "effective": effective
            })

            return fn(*args, **final_kwargs, **extras)
        return wrapper
    return decorator


def _ensure_dir(p: Path, reset: bool=False):
    if reset and p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

    
def figure_renderer(fig_name=None, reset=False, exts=(".png", ".svg")):
    def decorator(fn):
        # fn should returns "fig"
        module = sys.modules[fn.__module__]
        script_file = Path(module.__file__)
        prefix = Path(script_file.stem)
        out_dir = ROOT_DIR / prefix
        out_name = Path(fig_name)
        _ensure_dir(out_dir, reset)

        @log_params()
        @wraps(fn)
        def wrapper(*args, **kwargs):
            
            _out_name = kwargs.get("_func_label", None)
            if _out_name is None:
                _out_name = out_name
            else:
                _out_name = Path(kwargs.pop("_func_label"))
            fig = fn(*args, **kwargs)

            print("Figure save into", out_dir / _out_name)
            for ext in exts:
                fig.savefig(out_dir / _out_name.with_suffix(ext), bbox_inches="tight", transparent=False, dpi=300)
            
            # auto-save 
            param_tracker.calls[-1]["figure"] = str(_out_name)
            param_tracker.save(out_dir / prefix.with_suffix(".params.yml"))

        return wrapper
    return decorator


def save_fig(fig, filename_wo_ext):
    prefix = Path(__file__).stem
    fdir = os.path.join(ROOT_DIR, prefix)
    if not os.path.exists(fdir):
        print("Make directory: %s"%(fdir))
        os.makedirs(fdir)
    fig.savefig(os.path.join(fdir, f"{filename_wo_ext}.png"))
    fig.savefig(os.path.join(fdir, f"{filename_wo_ext}.svg"))
    
    
# def get_figure(figsize):
#     return plt.figure(figsize=figsize)


if __name__ == "__main__":
    
    @log_params()
    def test(a=2, b=3, c=None):
        print("Running:", a, b)

    test()

