import os

from src.trainer import process_image

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import torch
import numpy as np
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import segyio
import matplotlib.pyplot as plt
from PIL import Image

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)


def main():
    global model
############ Train ##############
    # args.test_only = False
    # args.save_results = False
    # checkpoint = utility.checkpoint(args)
    # loader = data.Data(args)
    # _model = model.Model(args, checkpoint)
    # _loss = loss.Loss(args, checkpoint)
    # _lossv = loss.Loss(args, checkpoint, m='validation')
    # t = Trainer(args, loader, _model, _loss, _lossv, checkpoint)
    # while not t.terminate():
    #     t.train()
    #     t.test()
    # checkpoint.done()

# ############ Test synthetic data ##############
#     print('test synthetic data')
#     args.test_only = True
#     args.save_results = True
#     args.pre_train = '../experiment/alpha6/model/model_best.pt'
#     args.data_range = '1-1200/1451-1600'
#     checkpoint = utility.checkpoint(args)
#     loader = data.Data(args)
#     _model = model.Model(args, checkpoint)
#
#     t = Trainer(args, loader, _model, ckp=checkpoint)
#     t.test()
#
#     checkpoint.done()

########### Test2 ##############
    print("test field data")
    args.test_only = True
    args.save_dir_suffix = 'field'
    args.data_range = '1-1200/1451-1453'
    args.dir_lr = '../data/field/'
    args.apply_field_data = True
    args.pre_train = '../experiment/alpha6/model/model_best.pt'
    checkpoint = utility.checkpoint(args)
    loader = data.Data(args)
    _model = model.Model(args, checkpoint)

    t = Trainer(args, loader, _model, ckp=checkpoint)
    # p = np.loadtxt('../experiment/alpha6/model/0000.dat')
    # print()
    #
    # process_image(t, p, '../data/nx2/00.png')

    t.test()
    # t.deme_lovto()
    checkpoint.done()


def read_segy_data(segy_file):
    # segy = segyio.open(segy_file)
    # data = segyio.tools.cube(segy)
    with segyio.open(segy_file, "r") as segyfile:
        traces = segyfile.trace.raw[:]
    return traces


if __name__ == '__main__':
    # Load the SEGY file
    # segy_file = "../data/field/sliced_seismic_data4.segy"
    # data = read_segy_data(segy_file)
    #
    # # Convert to numpy array and slice to 300x200
    # patch = np.array(data)[200:500, 200:400].T  # Ensure this gives the desired shape (300x200)
    #
    # # Convert to an image using PIL
    # sr_img_array = Image.fromarray(patch, mode='L')
    # # Save as NPY file
    # np.save("../data/field/output.npy", patch)
    # # Save as image
    # plt.figure(figsize=(10, 6))
    # plt.imshow(patch, cmap='seismic', aspect='auto')
    # plt.colorbar(label='Amplitude')
    # plt.title('Seismic Patch')
    # plt.xlabel('Trace')
    # plt.ylabel('Time/Depth')
    # plt.savefig("300_200.png", dpi=100, bbox_inches='tight')
    # plt.close()


    # # Visualization of raw data
    # # fig = plt.figure(figsize=(18, 9))
    # plt.imshow(data.T, cmap='gray')
    # plt.gca().invert_xaxis()
    # # fig.patch.set_alpha(0)
    # plt.gca().patch.set_alpha(0)
    # plt.tight_layout()
    # plt.savefig(f"n-segy.png")
    # plt.close()

    # data = np.array(data, dtype=np.float32)
    # output_file = "../data/field/american_egret_npy/output.npy"
    # np.save(output_file, data)
    main()
    # image = np.load("../data/field/american_egret_npy/output.npy")
    # sr = model.Model.model(image)

    # CHECK
    # my_data = np.load('../data/field/output.npy')
    # their_data = np.fromfile("../data/field\lulia_592x400.dat", dtype=np.float32)
    # print(my_data.shape, my_data.min(), my_data.max())
    # print(their_data.shape, their_data.min(), their_data.max())


