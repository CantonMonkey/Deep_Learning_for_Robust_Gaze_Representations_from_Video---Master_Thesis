import torch
from Integrated_model import WholeModel


def test():
    model = WholeModel()

    bs = 16 # batch size
    # hidden_size =
    # sequence_length =
    # num_classes =

    Leye = torch.rand(bs,3, 16, 16)
    Reye = torch.rand(bs,3, 16, 16)
    FaceData = torch.rand(bs,3, 16, 16)  # currently matched with eyes......

    out = model(Leye, Reye, FaceData)
    print(out.shape())


if __name__ == "__main__":
    test()




