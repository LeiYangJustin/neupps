import igl
import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def eig_decomp(v, f):
    l = -igl.cotmatrix(v, f)
    m = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)

    k = 10
    d, u = sp.sparse.linalg.eigsh(l, k, m, sigma=0, which="LM")

    u = (u - np.min(u)) / (np.max(u) - np.min(u))
    bbd = 0.5 * np.linalg.norm(np.max(v, axis=0) - np.min(v, axis=0))

    print(bbd.shape)

    # p = subplot(v, f, bbd * u[:, 0], shading={"wireframe":False, "flat": False}, s=[1, 2, 0])
    # subplot(v, f, bbd * u[:, 1], shading={"wireframe":False, "flat": False}, s=[1, 2, 1], data=p)


def fft2d(uv, yimage):

    yimage = yimage.reshape(yimage.shape[0], yimage.shape[1], -1)

    dim = yimage.shape[2]
    for i in range(dim):
        plt.cla()

        average = np.mean(yimage[:, :, i])
        std = np.std(yimage[:, :, i])
        yimage[:, :, i] = yimage[:, :, i] - average
        fft = np.fft.fft2(yimage[:, :, i])
        fft_shift = np.fft.fftshift(fft)

        # Calculate the magnitude spectrum
        magnitude_spectrum = 20 * np.log(np.abs(fft_shift))

        # Display the original image and the magnitude spectrum
        plt.subplot(121), plt.imshow(yimage[:, :, i], cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        plt.show()


def fft1d(u, yimage, stop_freq=None, quantiles=[0.9999], save_name=None):

    ## subtract the mean
    yimage= yimage - np.mean(yimage)
    yimage = yimage / np.std(yimage)

    # Generate a test signal
    fft = np.fft.rfft(yimage)
    fft_freq = np.fft.rfftfreq(len(yimage), u[1] - u[0])

    # Calculate the magnitude spectrum
    magnitude_spectrum = np.abs(fft)
    stop_freq = stop_freq

    if stop_freq is not None:
        fft_freq = fft_freq[:stop_freq]
        magnitude_spectrum = magnitude_spectrum[:stop_freq]**2
    sum_mag = np.sum(magnitude_spectrum)
    magnitude_spectrum = magnitude_spectrum / sum_mag

    cumsum_mag = np.cumsum(magnitude_spectrum)
    # print(cumsum_mag)
    ## find the frequency that contains 99% of the energy
    cutoff_freqs = []
    for q in quantiles:
        mask = cumsum_mag > q
        ## find the first bool true
        cutoff_freqs.append(np.argmax(mask))

    # Display the signal and the magnitude spectrum
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(u, yimage)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Amplitude')
    print(fft_freq.shape, magnitude_spectrum.shape)
    print(fft_freq)
    # ax2.stem(fft_freq[:400], np.sqrt(magnitude_spectrum[:400]), markerfmt=" ", basefmt="")
    ax2.plot(fft_freq[:400], np.sqrt(magnitude_spectrum[:400]))
    for cf in cutoff_freqs:
        ax2.stem([fft_freq[cf]], [np.sqrt(magnitude_spectrum.max())], markerfmt="r-", linefmt='r')
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Magnitude')
    if save_name is not None:
        plt.savefig(save_name, dpi=300)
        plt.close()
    else:
        plt.show()
