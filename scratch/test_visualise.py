import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import os
from omegaconf import OmegaConf
from afdad.graph.workflow import build_graph

# Setup basic hydra/omegaconf config
# We want to mimic the configuration Hydra passes to main.
# Let's import hydra compose API to construct it properly!
import hydra
from hydra import initialize, compose

try:
    print("Initialising Hydra compose...")
    with initialize(version_base=None, config_path="../afdad/configs"):
        cfg = compose(config_name="config", overrides=["mode=visualise"])
    print("Config composed successfully.")
    
    print("Calling build_graph(cfg)...")
    compiled = build_graph(cfg)
    print("build_graph(cfg) returned successfully.")
    
    print("Calling compiled.get_graph()...")
    graph = compiled.get_graph()
    print("get_graph() returned successfully.")
    
    print("Calling draw_mermaid_png()...")
    png_data = graph.draw_mermaid_png()
    print(f"draw_mermaid_png() returned successfully, data size: {len(png_data)}")
    
except Exception as e:
    import traceback
    print("An exception occurred:")
    traceback.print_exc()
    sys.exit(1)
