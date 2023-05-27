from transformers import AutoTokenizer, AutoModel
import os
import platform
import signal

'''
run in linux
python glm.py
'''


def debug():
    print("start......")
    model_path = "./data/chatglm-6b-int4"
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).float()
    print("model loaded")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("tokenizer load")
    model = model.eval()
    response, history = model.chat(tokenizer, "你好", history=[])
    print(response)


def build_prompt(history):
    prompt = "欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，输入stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
    return prompt


def main():
    print("start......")
    model_path = "./data/chatglm-6b-int4"
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).float()
    print("model loaded")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("tokenizer load")
    model = model.eval()

    history = []
    print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，输入stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        for response, history in model.stream_chat(tokenizer, query, history=history):
            print(build_prompt(history), flush=True)
        print(build_prompt(history), flush=True)


if __name__ == '__main__':
    main()
