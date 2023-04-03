from transformers import AutoTokenizer, AutoModel


def main():
    model_path = "THUDM/chatglm-6b-int4"
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).float()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model.eval()
    response, history = model.chat(tokenizer, "你好", history=[])
    print(response)


if __name__ == '__main__':
    main()
