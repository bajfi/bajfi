import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
import os

def list_files(root,suffix=''):
    for root,dirs,files in os.walk(folder_path):
        for file in files:
            if file.endswith(suffix):
                yield (Path(root) / file)
        for dir_ in dirs:
            list_files(Path(root) / dir_)

def db(rms):
    return 20*np.log10(rms/2e-5)

def draw(times,diff):
    fig, axs = plt.subplots(2, 1, 
        constrained_layout=True,
        figsize=(10,8))
    axs[0].plot(times,diff)
    axs[0].set_xlabel("Time ($s$)")
    axs[0].set_ylabel("Amplitute ($Pa$)")

    # fft transform
    sp = np.fft.fft(diff)
    freq = np.fft.fftfreq(sp.shape[0])
    axs[1].plot(freq,sp.real)
    axs[1].set_xlabel("Frequence ($Hz$)")
    axs[1].set_ylabel("Amplitute ($Pa$)")

    return fig

def file_process(file,csv_path='',img_path=''):
    data = []
    with open(file,'r') as f:
        # ignore the headers
        for line in f.readlines()[4:-1]:
            data.append(list(map(eval,line.split())))

    data = pd.DataFrame(data,columns=["time","pressure"])

    # calculate the difference
    # pressure - mean(pressure) 
    pressure_mean = data["pressure"].mean()
    data["diff"] = data["pressure"] - pressure_mean

    # calculate the overall level with RMS
    pressure_rms = np.sqrt((data["pressure"]**2).mean())

    # store db
    data["db"] = pd.Series([db(pressure_rms)],dtype=np.float64)

    # check is export the date to csv file
    if csv_path:
        # save the csv files in current folder
        csv_name = file.with_suffix(".csv").name
        saved_path = file.parent / csv_path / csv_name
        try:
            data.to_csv(saved_path)
        except Exception as e:
            print(e,"\nmaking directory ...")
            saved_path.parent.mkdir()
            data.to_csv(saved_path)

    # export fft result image
    if img_path:
        # save the image in current folder
        img_name = file.with_suffix(".png").name
        saved_path = file.parent / img_path / img_name

        fig = draw(data["time"],data["diff"])
        fig.suptitle(file.stem)
        try:
            fig.savefig(saved_path)
        except Exception as e:
            print(e,"\nmaking directory ...")
            saved_path.parent.mkdir()
            fig.savefig(saved_path)
        finally:
            plt.close(fig)


if __name__ == '__main__':
    folder_path = r"data"   # root path for search
    suffix = r".ard"        # processing file suffix
    csv_path = r"CSVfiles"  # saving csv folder name
    img_path = r"Figures"   # saving img folder name
    for file in tqdm(list_files(folder_path,suffix)):
        file_process(file,csv_path,img_path)
