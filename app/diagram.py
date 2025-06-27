# -*- coding: utf-8 -*-
# @file diagram.py
# @brief The Diagram Generation Application
# @author sailing-innocent
# @date 2025-02-28
# @version 1.0
# ---------------------------------

from app.base import AppConfigBase, AppBase 
import importlib 

class DiagramAppConfig(AppConfigBase):
    figures = {}
    def __init__(self, config_path):
        super().__init__() # load default config keys 
        self._config_keys += ["figures"]
        self.from_toml(config_path)
        print(self.figures)

    @property 
    def out_dir(self):
        return self.figures["outdir"]

    def get_figure(self, name: str):
        if name in self.figures.keys():
            return self.figures[name]
        return None

class DiagramApp(AppBase):
    def __init__(self, config_path="config/diagram.toml"):
        super().__init__(DiagramAppConfig(config_path))
        
    def run(self, name: str):
        print(f"Running DiagramApp with {name}")
        base_outdir = self.config.out_dir
        fig_config = self.config.get_figure(name)
        if fig_config is None:
            print(f"Figure {name} not found")
            return
        try:
            module = importlib.import_module(f"module.diagram.{fig_config['module']}")
            outdir = f"{base_outdir}/{fig_config['outdir']}"
            figname = fig_config["name"]
            res = module.draw(figname, outdir)
            if res:
                print(f"Success Draw! Saved {figname}.png to {outdir}")
        except Exception as e:
            print(f"Failed to run {name}: {e}")
    
