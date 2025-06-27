import json
import types
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import (
    TencentCloudSDKException,
)
from tencentcloud.lkeap.v20240522 import lkeap_client, models
import sys

import os 

class StreamRes:
    def __init__(self):
        self.reason = ""
        self.content = ""

    def apply_delta(self, delta):
        if "ReasoningContent" in delta:
            self.reason += delta["ReasoningContent"]
        if "Content" in delta:
            self.content += delta["Content"]

    def __str__(self):
        return f"reason: {self.reason}, content: {self.content}"

# wrapper with endpoint, id, key
def tencent_cloud_deepseek(endpoint, id, key):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(endpoint, id, key, *args, **kwargs)
        return wrapper
    return decorator



def request_prompt_impl(endpoint, id, key, prompt: str = "", model: str ="deepseek-r1") -> StreamRes:
    # prompt = "请问9.11和9.8哪个数字更大？"
    res = StreamRes()
    try:
        # 实例化一个认证对象，入参需要传入腾讯云账户 SecretId 和 SecretKey，此处还需注意密钥对的保密
        # 代码泄露可能会导致 SecretId 和 SecretKey 泄露，并威胁账号下所有资源的安全性。以下代码示例仅供参考，建议采用更安全的方式来使用密钥，请参见：https://cloud.tencent.com/document/product/1278/85305
        # 密钥可前往官网控制台 https://console.cloud.tencent.com/cam/capi 进行获取

        # cred = credential.Credential("SecretId", "SecretKey")
        cred = credential.Credential(id, key)
        # 实例化一个http选项，可选的，没有特殊需求可以跳过
        httpProfile = HttpProfile()
        httpProfile.endpoint = endpoint
        # httpProfile.reqTimeout = 60 # 请求超时时间，单位为秒（默认60秒）

        # 实例化一个client选项，可选的，没有特殊需求可以跳过
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        # 实例化要请求产品的client对象,clientProfile是可选的
        client = lkeap_client.LkeapClient(cred, "ap-shanghai", clientProfile)

        # 实例化一个请求对象,每个接口都会对应一个request对象
        req = models.ChatCompletionsRequest()
        params = {
            "Model": model,
            "Messages": [{"Role": "user", "Content": prompt}],
            "Stream": True,
            "Temperature": 0.5,
        }
        req.from_json_string(json.dumps(params))

        # 返回的resp是一个ChatCompletionsResponse的实例，与请求对象对应
        resp = client.ChatCompletions(req)

        # 输出json格式的字符串回包
        if isinstance(resp, types.GeneratorType):  # 流式响应
            for event in resp:
                if event["data"] == "[DONE]":
                    break
                if event["data"] == "":
                    continue
                # print(event["data"])
                res_json = json.loads(event["data"])
                # print(res["Choices"][0]["Delta"])
                delta = res_json["Choices"][0]["Delta"]
                res.apply_delta(delta)
                # sys out the res_json
                sys.stdout.write('\r')

                sys.stdout.write(json.dumps(delta, ensure_ascii=False))
                sys.stdout.flush()

        else:  # 非流式响应
            print(resp)

    except TencentCloudSDKException as err:
        print(err)

    return res 

@tencent_cloud_deepseek(os.getenv("TENCENT_CLOUD_DEEPSEEK_ENDPOINT"),
                        os.getenv("TENCENT_CLOUD_DEEPSEEK_ID"),
                        os.getenv("TENCENT_CLOUD_DEEPSEEK_KEY"))
def request_prompt(endpoint, id, key, prompt: str = "", model: str = "deepseek-r1") -> StreamRes:
    """
    Request a prompt from Tencent Cloud DeepSeek.

    Args:
        endpoint (str): The endpoint for the Tencent Cloud DeepSeek API.
        id (str): The Tencent Cloud ID.
        key (str): The Tencent Cloud Key.
        prompt (str): The prompt to send to the model.
        model (str): The model to use, default is "deepseek-r1".

    Returns:
        StreamRes: The response containing reasoning and content.
    """
    return request_prompt_impl(endpoint, id, key, prompt, model)


def run(prompt: str = "请问9.11和9.8哪个数字更大？", model: str = "deepseek-r1"):
    """
    Run the request prompt function with the given prompt and model.

    Args:
        prompt (str): The prompt to send to the model.
        model (str): The model to use, default is "deepseek-r1".
    """
    res = request_prompt(prompt, model)
    print(f"\nResponse:\n{res}")