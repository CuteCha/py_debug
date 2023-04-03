def search01(arr, start, end):
    if start >= end:
        return arr[start]

    mid = (start + end) // 2

    if arr[mid] > arr[end]:
        return search01(arr, mid + 1, end)
    elif arr[mid] > arr[start]:
        return search01(arr, mid, end)
    else:
        return search01(arr, start, mid)


def search(arr, start, end):
    if start >= end:
        return arr[start]

    mid = (start + end) // 2
    if arr[mid] > arr[end]:
        return search(arr, mid + 1, end)
    else:
        return search(arr, start, mid)


def main():
    arr = [5, 7, 0, 1, 3]

    print(search(arr, 0, len(arr) - 1))


if __name__ == '__main__':
    main()
