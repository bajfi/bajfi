import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.fft import rfft, rfftfreq


# get the required file recursively
def list_files(path, suffix=''):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(suffix):
                yield Path(root) / file
        for dir_ in dirs:
            list_files(Path(root) / dir_, suffix)


def db(rms):
    return 20 * np.log10(rms / 2e-5)


def draw(data):
    fig, axs = plt.subplots(2, 1,
                            constrained_layout=True,
                            figsize=(10, 8))
    times = data[['time']].to_numpy().squeeze()
    diff = data[['diff']].to_numpy().squeeze()
    N = times.shape[0]
    axs[0].plot(times, diff)
    axs[0].set_xlabel("Time ($s$)")
    axs[0].set_ylabel("Amplitute ($Pa$)")

    # fft transform
    sp = rfft(diff)
    freq = rfftfreq(N, times[1] - times[0])
    axs[1].plot(freq[1:N // 2], 2 * np.abs(sp[1:N // 2]) / N)
    axs[1].set_xlabel("Frequence ($Hz$)")
    axs[1].set_ylabel("Amplitute ($Pa$)")

    # save frequency and sp to data
    data['freq'] = pd.Series(freq[:N // 2])
    data['amplitute'] = pd.Series(2 * np.abs(sp[:N // 2]) / N)

    return fig


def file_process(file, csv_path='', img_path=''):
    data = []
    with open(file, 'r') as f:
        # ignore the headers
        for line in f.readlines()[4:-1]:
            data.append(list(map(eval, line.split())))

    data = pd.DataFrame(data, columns=["time", "pressure"], dtype=np.float64)

    # calculate the difference
    # pressure - mean(pressure)
    pressure_mean = data["pressure"].mean()
    data["diff"] = data["pressure"] - pressure_mean

    # calculate the overall level with RMS
    pressure_rms = np.sqrt((data["diff"] ** 2).mean())

    # store db
    data["db"] = pd.Series([db(pressure_rms)], dtype=np.float64)

    # export fft result image
    if img_path:
        img_name = file.with_suffix(".png").name
        saved_path = file.parent / img_path / img_name

        fig = draw(data)
        fig.suptitle(file.stem)
        try:
            fig.savefig(saved_path)
        except Exception as e:
            print(e, "\nmaking directory ...")
            saved_path.parent.mkdir()
            fig.savefig(saved_path)
        finally:
            plt.close(fig)

    # check is export the date to csv file
    if csv_path:
        # make a dir to store the csv file
        csv_name = file.with_suffix(".csv").name
        saved_path = file.parent / csv_path / csv_name
        try:
            data.to_csv(saved_path)
        except Exception as e:
            print(e, "\nmaking directory ...")
            saved_path.parent.mkdir()
            data.to_csv(saved_path)


if __name__ == '__main__':
    # 把这里换成数据文件夹
    folder_path = r"2"  # 需要遍历的根目录（只需要改这里）
    csv_path = r"CSVfiles"  # 保存csv文件的文件夹名
    img_path = r"Figures"  # 保存图片的文件夹名
    suffix = r".ard"  # 需要得到文件的后缀名

    for file in tqdm(list_files(folder_path, suffix), desc="processing: "):
        file_process(file, csv_path, img_path)
    print("done")
