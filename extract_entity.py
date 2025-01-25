# importing the module
import re
import jionlp


def extract_ip():
    pattern = re.compile('''((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)''')

    regex_str = '''((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'''
    text = '''000.0000.00.00
    192.168.1.1
    912.465.123.123
    192.168.4.164
    69.168.4.226
    32.89.31.164
    67.168.3.227'''

    res = pattern.search(text)
    print(res)
    print(jionlp.extract_ip_address(text, detail=False))
    print(jionlp.extract_ip_address(text, detail=True))
    print(text[19:30])


if __name__ == '__main__':
    extract_ip()
