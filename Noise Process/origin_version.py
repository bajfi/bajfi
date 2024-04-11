import os
import sys
from pathlib import Path

import numpy as np
from numpy import ndarray
import pandas as pd
from scipy.fft import rfft, rfftfreq
import originpro as op

os.system(os.path.join(op.path("e"), "Origin.exe"))


# Very useful, especially during development, when you are
# liable to have a few uncaught exceptions.
# Ensures that the Origin instance gets shut down properly.
# Note: only applicable to external Python.
def origin_shutdown_exception_hook(exctype, value, traceback):
    """
    Ensures Origin gets shut down if an uncaught exception
    """
    op.exit()
    sys.__excepthook__(exctype, value, traceback)


if op and op.oext:
    sys.excepthook = origin_shutdown_exception_hook

# Set Origin instance visibility.
# Important for only external Python.
# Should not be used with embedded Python.
if op.oext:
    op.set_show(True)


# get the required file recursively
def list_files(path, suffix=""):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(suffix):
                yield Path(root) / file
        for dir_ in dirs:
            list_files(Path(root) / dir_, suffix)


def db(rms):
    return 20 * np.log10(rms / 2e-5)


def file_process(file:str,lastN:int):
    """
    use FFT the transform the single file's data
    """
    data = []
    with open(file, "r") as f:
        if file.endswith(r'.ard'):
            # ignore the headers
            for line in f.readlines()[4:-1]:
                data.append(list(map(eval, line.split())))
        elif file.endswith(r'.out'):
            for line in f.readlines()[3:-1]:
                _,value,t = map(eval, line.split())
                data.append([t,value])
        else:
            raise NotImplementedError
            
    # only need the last N numbers
    data = pd.DataFrame(data[-lastN:], columns=["time", "pressure"], dtype=np.float_)

    # calculate the difference
    # pressure - mean(pressure)
    pressure_mean = data["pressure"].mean()
    data["diff"] = data["pressure"] - pressure_mean

    # fft transform
    diff:ndarray = data[["diff"]].to_numpy().squeeze()
    times:ndarray = data[["time"]].to_numpy().squeeze()
    N:int = data.shape[0]
    sp = rfft(diff)
    freq = rfftfreq(N, times[1] - times[0])

    # save frequency and sp to data
    data["freq"] = pd.Series(freq[: N])
    data["amplitute"] = pd.Series(2 * np.abs(sp[: N]) / N)

    # calculate the overall level with RMS
    pressure_rms = np.sqrt((data["diff"] ** 2).mean())
    # store db
    data["db"] = pd.Series([db(pressure_rms)], dtype=np.float64)

    return data


def folder_process(
    folder,
    file_suffix,
    File_name="Noise Process",
    target_folder=op.path("u"),
    lastN:int=200,
    draw: bool = False,
):
    pre_book_name = ""
    wb = op.new_book("w")

    try:
        for file in list_files(folder, file_suffix):
            file_name = file.stem
            parant_name = file.parent.name
            if parant_name != pre_book_name:
                # make a new workbook
                print(f"make new book -- {parant_name}")
                pre_book_name = parant_name
                wb = op.new_book("w", parant_name)
                wb.name = parant_name
                wb.comments = f"the data group in the {parant_name} folder"
                # we don't use the sheet1,so we can delete it
                sheet1 = op.find_sheet("w", f"[{parant_name}]Sheet1")
                if sheet1 is not None:
                    sheet1.destroy()
            # add data to worksheet
            ws = wb.add_sheet(file_name)
            data = file_process(str(file.resolve()),lastN=lastN)
            ws.from_df(data)
            ws.set_labels(["s", "Pa", "Pa", "Hz", "Pa", "db"], type_="U")
            # plot fft transform if needed
            if draw:
                graph = op.new_graph(parant_name + "-" + file_name, template="line")
                graph[0].add_plot(ws, coly="E", colx="D")
                graph[0].rescale()
                graph[0].set_xlim(-100)
                
    except Exception as e:
        raise e
    finally:
        # we don't use the Book1,so we can delete it
        book1 = op.find_book("w", "Book1")
        if book1 is not None:
            book1.destroy()

        # save the project and shut down the application
        op.save(os.path.join(target_folder, Path(File_name).stem + ".opju"))
        if op.oext:
            op.exit()
        print("done !!\n")
        print(f"file has been {Path(target_folder).resolve()} !!!")


if __name__ == "__main__":
    # change the folder here
    folder_path = r"..."
    File_name = r"Noise Process"  # file name to save
    target_folder = op.path('u')  # folder name to save
    suffix = r".out"  # the suffix of the target file
    lastN:int = 1000    # last N value need to evaluate

    folder_process(
        folder_path,
        suffix,
        File_name=File_name,
        target_folder=target_folder,
        lastN=lastN,
        draw=True,
    )
