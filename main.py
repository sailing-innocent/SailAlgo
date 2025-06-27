# -*- coding: utf-8 -*-
# @file main.py
# @brief The Main Application Entry
# @author sailing-innocent
# @date 2025-02-28
# @version 1.0
# ---------------------------------

import argparse 
from app.diagram import DiagramApp

def main():
    parser = argparse.ArgumentParser(description="The Main Application Entry")
    parser.add_argument("-c", "--config", help="The configuration file path", default="config/diagram.toml")
    parser.add_argument("-a", "--app_type", help="The application type", default="diagram")
    parser.add_argument("-t", "--task", help="The task to run", default="all")
    args = parser.parse_args()
    print(f"Running with config: {args.config}")
    if args.app_type == "diagram":
        app = DiagramApp(args.config)
    else:
        raise ValueError(f"Unknown app_type: {args.app_type}")
    assert app.valid()
    app.run(args.task)

if __name__ == "__main__":
    main()