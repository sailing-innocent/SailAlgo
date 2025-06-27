# -*- coding: utf-8 -*-
# @file image_classify.py
# @brief The Image Classify App
# @author sailing-innocent
# @date 2025-03-11
# @version 1.0
# ---------------------------------

from app.base import AppConfigBase, AppBase 
import importlib 




# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mission", type=str, default="image_classify")
#     parser.add_argument("--test", action="store_true")
#     parser.add_argument("--train", action="store_true")
#     args = parser.parse_args()
#     assert args.test or args.train, "Specify at least one in --test or --train"

#     if args.mission == "image_classify":
#         from mission.image_classify import batch_run
#         batch_run(test_only=args.test)
#     else:
#         raise ValueError("Unknown mission: %s" % args.mission)