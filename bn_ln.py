import torch
from torch.nn import BatchNorm1d
from torch.nn import LayerNorm


def main():
    x = torch.tensor([[4.0, 3.0, 2.0],
                      [3.0, 3.0, 2.0],
                      [2.0, 2.0, 2.0]
                      ])

    print(BatchNorm1d(x.size()[1])(x))
    print(LayerNorm(x.size()[1:])(x))
    print("=" * 36)

    y = torch.tensor([
        [[1.0, 4.0, 7.0],
         [0.0, 2.0, 4.0]
         ],
        [[1.0, 3.0, 6.0],
         [2.0, 3.0, 1.0]
         ]
    ])
    print((BatchNorm1d(2))(y))
    print((LayerNorm(3))(y))


if __name__ == '__main__':
    main()
