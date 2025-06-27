# -*- coding: utf-8 -*-
# @file base.py
# @brief The Base Application
# @author sailing-innocent
# @date 2025-02-28
# @version 1.0
# ---------------------------------

import os 
import toml
from typing import Dict, Any

class AppConfigBase:
    _valid: bool = False
    
    def __init__(self, name: str = "", version: str = ""):
        self.name = name
        self.version = version
        self._valid = False
        self._config_keys = ["name", "version"]

    def __str__(self):
        return f"{self.name} v{self.version}"
        
    def from_toml(self, tpath):
        """Load configuration from TOML file
        
        Args:
            tpath (str): Path to TOML file
        
        Returns:
            bool: True if successfully loaded, False otherwise
        """
        if not os.path.exists(tpath):
            print(f"Config file not found: {tpath}")
            return False
        
        try:
            config_dict = toml.load(tpath)
            
            # Update instance attributes from TOML
            for key in self._config_keys:
                if key in config_dict:
                    setattr(self, key, config_dict[key])
            
            self._valid = True
            return True
        except Exception as e:
            print(f"Error loading TOML file: {e}")
            self._valid = False
            return False

    def to_toml(self, tpath):
        """Save configuration to TOML file
        
        Args:
            tpath (str): Path to save TOML file
        
        Returns:
            bool: True if successfully saved, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(tpath)), exist_ok=True)
            
            # Convert to dict
            config_dict = {key: getattr(self, key) for key in self._config_keys}
            
            # Write dict to TOML file
            with open(tpath, 'w') as f:
                toml.dump(config_dict, f)
            
            self._valid = True
            return True
        except Exception as e:
            print(f"Error saving TOML file: {e}")
            self._valid = False
            return False
    
    def update(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary
        
        Args:
            config_dict: Dictionary with configuration values
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            for key, value in config_dict.items():
                if key in self._config_keys:
                    setattr(self, key, value)
            self._valid = True
            return True
        except Exception as e:
            print(f"Error updating configuration: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary
        
        Returns:
            Dict: Configuration as dictionary
        """
        return {key: getattr(self, key) for key in self._config_keys}


class AppBase:
    def __init__(self, config: AppConfigBase):
        self.config = config
    
    def run(self):
        if not self.config._valid:
            print("Invalid configuration")
            return
        
        print(f"Running {self.config}")
        self._run()
    
    def _run(self):
        raise NotImplementedError
    
    def valid(self):
        return self.config._valid