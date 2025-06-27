import dotenv 
dotenv.load_dotenv()

# from samples.API.tencent_cloud_deepseek import run
# from samples.API.aliyun_deepseek import run
from samples.API.deepseek import run

import os 

if __name__ == "__main__":
    run()
