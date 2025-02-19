from decimal import Decimal
import src.utility
import torch
import torch.nn.utils as utils
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import src.data.common as common
import pandas as pd
from PIL import Image

class Trainer():
    def __init__(self, args, loader, my_model,
                 my_loss=None, my_lossv=None, ckp=None):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp  # ckp: checkpoint
        self.loader_train = loader.loader_train#
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.lossv = my_lossv
        self.optimizer = src.utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        device = torch.device('cpu')

        timer_data, timer_model = src.utility.timer(), src.utility.timer()
        # TEMP
        for batch, (lr, hr, _, _1) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            print("running")
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr.to(device))
            loss = self.loss(sr, hr)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1)
        )
        self.model.eval()
        if not self.args.test_only:
            self.lossv.start_log()

        scale = self.args.scale
        timer_test =src.utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        # self.deme_lovto()
        for (lr, hr, filename, params) in tqdm(self.loader_test, ncols=80):
            lr, hr = self.prepare(lr, hr)
            sr = self.model(lr)
            # Assuming `lr` and `sr` are single-channel images for mode 'L'
            lr_img_array = lr.cpu().numpy()[0][0]  # Get the first channel if multi-channel
            sr_img_array = sr.cpu().numpy()[0][0]

            # Convert arrays to images
            # lr_img = np.fromfile()


            # Assuming lr_img_array, sr_img_array, and hr_img_array are numpy arrays with values in the range [0, 1]
            lr_img = Image.fromarray((lr_img_array * 255).astype(np.uint8), mode='L')
            sr_img = Image.fromarray((sr_img_array * 255).astype(np.uint8), mode='L')
            # lr_img.show()
            # sr_img.show()
            lr_img.save(f"{filename}lr.png")
            sr_img.save(f"{filename}sr.png")

            # Convert PIL images to numpy arrays for plotting
            lr_img_array = np.array(lr_img)
            sr_img_array = np.array(sr_img)
            np.save(f"{filename}lr.npy", lr_img_array)

            # Create a figure and a set of subplots
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            # Plot each image
            axs[0].imshow(lr_img_array, cmap='gray')
            axs[0].set_title('Low Resolution (LR)')
            axs[0].axis('off')  # Hide axis

            axs[1].imshow(sr_img_array, cmap='gray')
            axs[1].set_title('Super Resolution (SR)')
            axs[1].axis('off')  # Hide axis


            # Display the plot
            plt.tight_layout()
            plt.savefig(f"{filename}.png")

            plt.close()
            if not self.args.test_only:
                lossv = self.lossv(sr, hr[1])

            save_list = [sr]
            if not self.args.apply_field_data:
                self.ckp.log[-1] += src.utility.calc_psnr(
                    sr, hr, scale
                )

            if self.args.save_results:
                self.ckp.save_results(filename[0], save_list, params)

        # self.deme_lovto()
        if not self.args.apply_field_data:
            self.ckp.log[-1] /= len(self.loader_test)
            best = self.ckp.log.max(0)
            self.ckp.write_log(
                '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                    'Seis',
                    scale,
                    self.ckp.log[-1],
                    best[0],
                    best[1] + 1
                )
            )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')
        if not self.args.test_only:
            self.lossv.end_log(len(self.loader_test))

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu')
        return [a.to(device) for a in args]

    def terminate(self):
        epoch = self.optimizer.get_last_epoch() + 1
        return epoch >= self.args.epochs

    def deme_lovto(self):
        # self.model.eval()

        # Load and prepare the image
        image = np.load("../data/field/output.npy").astype(np.float32)
        # image = np.load("../data/field/american_egret_npy/output.npy").astype(np.float32)
        # image = torch.from_numpy(image).float()
        image, _ = common.normal(image, image)
        image = torch.from_numpy(image[0])
        # Ensure the image has 4 dimensions [batch_size, channels, height, width]
        if image.ndim == 2:  # Single channel 2D image
            image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

        # Randomly generated high-resolution image
        hr = torch.from_numpy(np.random.rand(256, 256).astype(np.float32)).float()
        hr = hr.unsqueeze(0).unsqueeze(0)  # Ensuring hr is also 4D

        # Prepare low-resolution and high-resolution inputs
        lr, hr = self.prepare(image, hr)

        # Fix the shape of lr to 4D if it's 2D
        if lr.dim() == 2:  # lr is 2D, needs to be 4D
            lr = lr.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

        # Pass through the model
        print("DEMI lr: shape: ", lr.shape, " min: ", torch.min(lr), " max: ", torch.max(lr))
        sr = self.model(lr)
        print("DEMI sr: shape: ", sr.shape, " min: ", torch.min(sr), " max: ", torch.max(sr))
        # lr_img_array = lr.cpu().numpy()[0][0]
        # Assuming 'sr' is a tensor, visualize it using matplotlib
        sr_img = sr.cpu().numpy()[0][0]
        # sr_img = sr.squeeze().detach().cpu().numpy()
        show_save_two_images(lr,sr)

        # # plt.imshow(sr.squeeze().detach().cpu().numpy(), cmap='gray')   # Adjust if needed
        # plt.savefig("output_image.png")  # Save the image to a file
        # plt.close()  # Close the plot to free up resources


import torch
from PIL import Image
import torchvision.transforms as transforms


def inference_single_image(model, image_path):
    device = torch.device('cpu')
    model.eval()

    # Load and preprocess image
    img = Image.fromarray(image_path)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    lr_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Inference
    with torch.no_grad():
        lr_tensor = lr_tensor.to(device)
        sr_tensor = model(lr_tensor)

    # Convert back to image
    to_pil = transforms.ToPILImage()
    sr_image = to_pil(sr_tensor.squeeze().cpu())

    return sr_image

def show_save_two_images(lr, sr, filename='output.png', mode='L'):
    # Assuming `lr` and `sr` are single-channel images for mode 'L'
    lr_img_array = lr.cpu().numpy()[0][0]  # Get the first channel if multi-channel
    sr_img_array = sr.cpu().numpy()[0][0]

    # Convert arrays to images
    # lr_img = np.fromfile()

    # Assuming lr_img_array, sr_img_array, and hr_img_array are numpy arrays with values in the range [0, 1]
    lr_img = Image.fromarray((lr_img_array * 255).astype(np.uint8), mode='L')
    sr_img = Image.fromarray((sr_img_array * 255).astype(np.uint8), mode='L')
    # lr_img.show()
    # sr_img.show()
    lr_img.save(f"{filename}lr.png")
    sr_img.save(f"{filename}sr.png")

    # Convert PIL images to numpy arrays for plotting
    lr_img_array = np.array(lr_img)
    sr_img_array = np.array(sr_img)
    np.save(f"{filename}lr.npy", lr_img_array)

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot each image
    axs[0].imshow(lr_img_array, cmap='gray')
    axs[0].set_title('Low Resolution (LR)')
    axs[0].axis('off')  # Hide axis

    axs[1].imshow(sr_img_array, cmap='gray')
    axs[1].set_title('Super Resolution (SR)')
    axs[1].axis('off')  # Hide axis

    # Display the plot
    plt.tight_layout()
    plt.savefig(f"{filename}.png")

    plt.close()


# Example usage:
def process_image(model_instance, input_image_path, output_image_path):
    sr_image = inference_single_image(model_instance.model, input_image_path)
    sr_image.save(output_image_path)


def show_fourier2d_2_images(data1, data2):
    fft1 = np.fft.fft2(data1)
    fft2 = np.fft.fft2(data2)

    # Compute the magnitude spectrum (amplitude)
    magnitude1 = np.abs(fft1)
    magnitude2 = np.abs(fft2)

    # Combine the two spectra side by side
    combined_spectrum = np.hstack((magnitude1, magnitude2))

    # Create frequency indices
    freq_u = np.fft.fftfreq(data1.shape[0])  # Frequency indices for rows
    freq_v = np.fft.fftfreq(data2.shape[1])  # Frequency indices for columns

    # Create a grid of frequency indices
    freq_u_grid, freq_v_grid = np.meshgrid(freq_u, freq_v, indexing="ij")

    # Flatten and create a 2D table
    rows = []
    for i in range(combined_spectrum.shape[0]):
        for j in range(combined_spectrum.shape[1]):
            rows.append({
                "u (row freq)": freq_u_grid[i % data1.shape[0], j % data1.shape[1]],
                "v (col freq)": freq_v_grid[i % data1.shape[0], j % data1.shape[1]],
                "Magnitude (Matrix 1)": magnitude1[i % data1.shape[0], j % data1.shape[1]],
                "Magnitude (Matrix 2)": magnitude2[i % data2.shape[0], j % data2.shape[1]]
            })

    table = pd.DataFrame(rows)

    # Save the table to CSV (optional)
    table.to_csv("combined_spectrum_table.csv", index=False)

    # Display first few rows of the table
    print(table.head())

    # Optional: Visualize the combined spectrum
    plt.imshow(np.log1p(combined_spectrum), cmap="viridis")
    plt.title("Log Combined Fourier Spectrum")
    plt.colorbar()
    plt.show()
