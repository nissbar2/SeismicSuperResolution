import os

import torch
import numpy as np
import src.utility as utility
import src.my_data as data
import src.my_model as my_model
import src.my_loss as my_loss
from src.option import args
from src.my_trainer import Trainer
import segyio
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
my_device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    global my_model
############ Train ##############
    args.test_only = False
    args.save_results = False
    args.pre_train = 'experiment/alpha6/model/model_best.pt'
    checkpoint = utility.checkpoint(args)
    loader = data.Data(args)
    _model = my_model.Model(args, checkpoint)
    _loss = my_loss.Loss(args, checkpoint)
    _lossv = my_loss.Loss(args, checkpoint, m='validation')
    t = Trainer(args, loader, _model, _loss, _lossv, checkpoint)
    while not t.terminate():
        t.train()
        t.test()
    checkpoint.done()

# ############ Test synthetic data ##############
#     print('test synthetic data')
#     args.test_only = True
#     args.save_results = True
#     args.pre_train = '../experiment/alpha6/model/my_model_weights.pt'
#     args.data_range = '1-1200/1451-1600'
#     checkpoint = utility.checkpoint(args)
#     loader = data.Data(args)
#     _model = my_model.Model(args, checkpoint)
#
#     t = Trainer(args, loader, _model, ckp=checkpoint)
#     t.test()
#
#     checkpoint.done()

########### Test2 ##############
    # print("test field data")
    # args.test_only = True
    # args.save_dir_suffix = 'field'
    # args.data_range = '1-1200/1451-1453'
    # args.dir_lr = '../data/field/'
    # args.apply_field_data = True
    # args.pre_train = '../experiment/alpha6/model/model_best.pt'
    # checkpoint = utility.checkpoint(args)
    # loader = data.Data(args)
    # _model = my_model.Model(args, checkpoint)
    # t = Trainer(args, loader, _model, ckp=checkpoint)
    # t.test()
    # return

    # p = np.loadtxt('../experiment/alpha6/model/0000.dat')
    # print()
    #
    # process_image(t, p, '../data/nx2/00.png')

    # t.test()
    # t.deme_lovto()
    # checkpoint.done()


def read_segy_data(segy_file):
    # segy = segyio.open(segy_file)
    # data = segyio.tools.cube(segy)
    with segyio.open(segy_file, "r") as segyfile:
        traces = segyfile.trace.raw[:]
    return traces


def show_fourier2d_2_images(matrix1, matrix2):
    fft1 = np.fft.fft2(matrix1)
    fft2 = np.fft.fft2(matrix2)

    # Shift the zero-frequency component to the center
    fft1_shifted = np.fft.fftshift(fft1)
    fft2_shifted = np.fft.fftshift(fft2)

    # Compute the magnitude spectrum
    magnitude1 = np.abs(fft1_shifted)
    magnitude2 = np.abs(fft2_shifted)

    # Log-transform the magnitudes for better visualization
    log_magnitude1 = np.log1p(magnitude1)
    log_magnitude2 = np.log1p(magnitude2)

    # Plot the Fourier spectra side by side
    plt.figure(figsize=(12, 6))

    # Spectrum of the first matrix
    plt.subplot(1, 2, 1)
    plt.imshow(log_magnitude1, cmap="viridis", extent=(-0.5, 0.5, -0.5, 0.5))
    plt.title("Fourier Spectrum: Matrix 1")
    plt.xlabel("Frequency (u)")
    plt.ylabel("Frequency (v)")
    plt.colorbar(label="Log Magnitude")

    # Spectrum of the second matrix
    plt.subplot(1, 2, 2)
    plt.imshow(log_magnitude2, cmap="viridis", extent=(-0.5, 0.5, -0.5, 0.5))
    plt.title("Fourier Spectrum: Matrix 2")
    plt.xlabel("Frequency (u)")
    plt.ylabel("Frequency (v)")
    plt.colorbar(label="Log Magnitude")

    # Show the plot
    plt.tight_layout()
    plt.savefig("fourier_spectrum_side_by_side.png", dpi=300)

def load_trained_model_1_to_3(pre_train_path):
    weights = torch.load(pre_train_path, map_location=my_device)
    conv_weight_key = "start.conv1.0.0.weight"
    conv_bias_key = "start.conv1.0.0.bias"
    print(f"Shape of {conv_weight_key}: {weights[conv_weight_key].shape}")
    old_weights = weights[conv_weight_key]
    new_weights = old_weights.repeat(1, 3, 1, 1) / 3
    weights[conv_weight_key] = new_weights
    if conv_bias_key in weights:
        weights[conv_bias_key] = weights[conv_bias_key].clone()
    # Save the updated weights
    print(f"Shape of {conv_weight_key}: {weights[conv_weight_key].shape}")
    new_checkpoint_path = "experiment/alpha6/model/my_model_weights.pt"
    torch.save(weights, new_checkpoint_path)

    print(f"Modified weights saved to {new_checkpoint_path}")


if __name__ == '__main__':
    # print(torch.cuda.is_available())
    # print(torch.cuda.device_count())
    # print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")
    main()
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
    # main()
    # image = np.load("../data/field/american_egret_npy/output.npy")
    # sr = model.Model.model(image)

    # CHECK
    # my_data = np.load('../data/field/output.npy')
    # their_data = np.fromfile("../data/field\lulia_592x400.dat", dtype=np.float32)
    # print(my_data.shape, my_data.min(), my_data.max())
    # print(their_data.shape, their_data.min(), their_data.max())
    # show_fourier2d_2_images(my_data,my_data)






