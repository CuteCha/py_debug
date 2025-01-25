import numpy as np


def add_table():
    for i in range(1, 10):
        for j in range(1, i + 1):
            print(f"{j}+{i}={j + i: <2}", end=" ")
        print()


def gen_add():
    for i in range(10):
        a = np.random.randint(50)
        b = np.random.randint(50)
        print(f"{a} + {b} =")


def gen_sub():
    for i in range(10):
        a = np.random.randint(1, 50)
        b = np.random.randint(a)
        print(f"{a} - {b} =")


def mul_table():
    for i in range(1, 10):
        for j in range(1, i + 1):
            print(f"{j}x{i}={j * i: <2}", end=" ")
        print()


def gen_mul():
    for i in range(10):
        a = np.random.randint(50)
        b = np.random.randint(50)
        print(f"{a} x {b} =")


def gen_div():
    cnt = 0
    while cnt < 10:
        a = np.random.randint(2, 90)
        b = np.random.randint(1, a)
        if a % b == 0:
            print(f"{a} ÷ {b} =")
            cnt += 1


def gen_latex():
    arr = [["& " for _ in range(1, 11)] for _ in range(1, 11)]
    for i in range(1, 10):
        for j in range(1, i + 1):
            # arr[i][j] = f"&${j}\\times{i}={j * i}$"
            arr[i][j] = f"&${j}+{i}={j * i}$"

    # print(arr)
    for i in range(1, len(arr)):
        print(" ".join(arr[i][1:]) + " \\\\")


def gen_multiplication():
    idx_arr = [[(j, i) for i in range(2, 10)] for j in range(11, 100)]
    arr = list()
    for row_idx in idx_arr:
        row_arr = list()
        for idx in row_idx:
            i, j = idx
            element = "&\\begin{tabular}{lr}\n" \
                      + f"& {i}" + "\\\\\n" \
                      + f"$\\times$ & {j}" + "\\\\\n" \
                      + "\\hline\n" + "&\n" + "\\end{tabular}"
            row_arr.append(element)
        arr.append(row_arr)
    with open("./data/multiplication_latex.txt", "a") as fr:
        for row_arr in arr:
            fr.write("\n".join(row_arr) + " \\\\" + "\n")
            print("\n".join(row_arr) + " \\\\")
            # print("\\hline")


def gen_multiplication2():
    idx_arr = [[(j, i) for i in range(2, 10)] for j in range(11, 100)]
    arr = list()
    for row_idx in idx_arr:
        row_arr = list()
        for idx in row_idx:
            i, j = idx
            element = "\\begin{tabular}{lr}\n" \
                      + f"& {i}" + "\\\\\n" \
                      + f"$\\times$ & {j}" + "\\\\\n" \
                      + "\\hline\n" + "&\n" + "\\end{tabular}"
            row_arr.append(element)
        arr.append(row_arr)
    with open("./data/multiplication_latex.txt", "w") as fr:
        for row_arr in arr:
            fr.write("\n".join(row_arr) + " \\\\" + "\n\\\\" + "\n")
            print("\n".join(row_arr) + " \\\\")
            # print("\\hline")


def _debug():
    j = 21
    i = 9
    a = "&\\begin{tabular}{lr}\n" \
        + f"& {j}" + "\\\\\n" \
        + f"$times$ & {i}" + "\\\\\n" \
        + "\\hline\n" + "&\n" + "\\end{tabular}"
    print(a)


def env_proc():
    import sys
    for line in sys.stdin:
        line = line.strip()
        k = line.index("=")
        print(f'export {line[:k + 1]}"{line[k + 1:]}"')


def main():
    # add_table()
    # print("-" * 72)
    # gen_add()
    # gen_sub()
    # print("-" * 72)
    # mul_table()
    # print("-" * 72)
    # gen_mul()
    # gen_div()
    # gen_latex()
    # _debug()
    gen_multiplication2()


if __name__ == '__main__':
    main()
