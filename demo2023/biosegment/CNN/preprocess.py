from json.tool import main
import shutil
from tokenize import String
import soundfile as sf
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import sys
import warnings
import argparse
import h5py
import logging
import librosa
from numpy import (
    log,
    exp,
    infty,
    zeros_like,
    vstack,
    zeros,
    errstate,
    finfo,
    sqrt,
    floor,
    tile,
    concatenate,
    arange,
    meshgrid,
    ceil,
    linspace,
)
from scipy.signal import lfilter
from .LFCC_pipeline import lfcc
import yaml

# configs - init
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def read_sigs(data: pd.DataFrame):
    """Reads audio signals from a pd.DataFrame containing the directories of
    .wav files, and optionally start and end points within the .wav files.

    Args:
        data: A pd.DataFrame created by prep_data().

    Returns:
        A pd.DataFrame with columns 'recording-id' and 'signal', or if segments were provided
        'utterance-id' and 'signal'. The 'signal' column contains audio as np.ndarrays.

    Raises:
        AssertionError: If a wav file is not 16k mono.
    """

    wavs = {}
    ret = []
    for i, j in zip(data["recording-id"].unique(), data["extended filename"].unique()):
        wavs[i], rate = sf.read(j)
        assert rate == 16000 and wavs[i].ndim == 1, f"{j} is not formatted in 16k mono."
    if "utterance-id" in data:
        for _, row in data.iterrows():
            ret.append(
                [
                    row["utterance-id"],
                    wavs[row["recording-id"]][
                        int(float(row["start"]) * rate) : int(float(row["end"]) * rate)
                    ],
                ]
            )
        return pd.DataFrame(ret, columns=["utterance-id", "signal"])
    else:
        for _, row in data.iterrows():
            ret.append([row["recording-id"], wavs[row["recording-id"]]])
        return pd.DataFrame(ret, columns=["recording-id", "signal"])


def save(data: pd.DataFrame, path: str):
    """Saves a pd.DataFrame to a .h5 file.

    Args:
        data: A pd.DataFrame for saving.
        path: The filepath where the pd.DataFrame should be saved.
    """

    warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
    data.to_hdf(path, key="data", mode="w")


def process_data_dir(path: str):
    """Function for processing Kaldi-style data directory containing wav.scp,
    segments (optional), and utt2spk (optional).

    Args:
        path: The path to the data directory.

    Returns:
        A tuple of pd.DataFrame in the format (wav_scp, segments, utt2spk), where
        pd.DataFrame contain data from the original files -- see docs for read_data_file().
        If a file is missing a null value is returned eg. a directory without utt2spk would
        return:
        (wav_scp, segments, None)

    Raises:
        FileNotFoundError: If wav.scp is not found.
    """

    files = [f for f in os.listdir(path) if os.path.isfile(f"{path}/{f}")]
    try:
        wav_scp = read_data_file(f"{path}/wav.scp")
        wav_scp.columns = ["recording-id", "extended filename"]
    except FileNotFoundError:
        print("ERROR: Data directory needs to contain wav.scp file to be processed.")
        raise
    if "segments" not in files:
        segments = None
    else:
        segments = read_data_file(f"{path}/segments")
        segments.columns = ["utterance-id", "recording-id", "start", "end"]
        segments[["start", "end"]] = segments[["start", "end"]].astype(float)
    if "utt2spk" not in files:
        utt2spk = None
    else:
        utt2spk = read_data_file(f"{path}/utt2spk")
        utt2spk.columns = ["utterance-id", "speaker-id"]
    return wav_scp, segments, utt2spk


def read_data_file(path: str):
    """Function for reading standard Kaldi-style text data files (eg. wav.scp, utt2spk etc.)

    Args:
        path: The path to the data file.

    Returns:
        A pd.DataFrame containing the enteries in the data file.

    Example:
        Given a file 'data/utt2spk' with the following contents:
        utt0    spk0
        utt1    spk1
        utt1    spk2

        Running the function yeilds:
        >>> print(read_data_file('data/utt2spk'))
                0       1
        0    utt0    spk0
        1    utt1    spk1
        2    utt2    spk2

    """

    with open(path, "r") as f:
        return pd.DataFrame([i.split() for i in f.readlines()], dtype=str)


def extract_signal(path):
    wav_scp, segments, utt2spk = process_data_dir(path)
    # check for segments file and process if found
    if segments is None:
        print("WARNING: Segments file not found, entire audio files will be processed.")
        wav_scp = wav_scp.merge(read_sigs(wav_scp))
        wav_scp = wav_scp.merge(utt2spk)
        return wav_scp
    else:
        data = wav_scp.merge(segments)
        data = data.merge(utt2spk)
        data = data.merge(read_sigs(data))
        return data


# def extract_mfcc(file, n_mfcc=16, n_fft=256, hop_length=128, n_mels = 40, delta_1 = False, delta_2 = False):
#     sig, sr = sf.read(file)
def extract_mfcc(
    sig,
    sr=16000,
    n_mfcc=16,
    n_fft=256,
    hop_length=128,
    n_mels=40,
    delta_1=False,
    delta_2=False,
):
    mfcc = librosa.feature.mfcc(
        y=sig, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    if not (delta_1 or delta_2):
        return mfcc.T

    feat = [mfcc]

    if delta_1:
        mfcc_delta_1 = librosa.feature.delta(mfcc, order=1)
        feat.append(mfcc_delta_1)

    if delta_2:
        mfcc_delta_2 = librosa.feature.delta(mfcc, order=2)
        feat.append(mfcc_delta_2)

    return np.vstack(feat).T


def Deltas(x, width=3):
    hlen = int(floor(width / 2))
    win = list(range(hlen, -hlen - 1, -1))
    xx_1 = tile(x[:, 0], (1, hlen)).reshape(hlen, -1).T
    xx_2 = tile(x[:, -1], (1, hlen)).reshape(hlen, -1).T
    xx = concatenate([xx_1, x, xx_2], axis=-1)
    D = lfilter(win, 1, xx)
    return D[:, hlen * 2 :]


def extract_lfcc(
    sig,
    order_deltas = 2,
    **kwargs
):
    # sig, fs = sf.read(file)
    # put VAD here, if wanted
    lfccs = lfcc(
        sig=sig,
        **kwargs
    ).T
    if order_deltas > 0:
        feats = list()
        feats.append(lfccs)
        for d in range(order_deltas):
            feats.append(Deltas(feats[-1]))
        lfccs = vstack(feats).T
    return lfccs


def extract_features(sig, features, cached=False, **kwargs):
    def get_feats():
        if features == "lfcc":
            return extract_lfcc(sig, **kwargs)
        elif features == "mfcc":
            return extract_mfcc(sig, **kwargs)
        else:
            return None

    print("================{}================".format(sig["extended filename"]))
    sig = sig["signal"]
    if cached:
        # cqcc is very slow, writing entire dataset to hdf5 file beforehand (offline cache)
        cache_file = features + ".h5"
        h5 = h5py.File(cache_file, "a")
        # group = h5.get(cache_file)
        # if group is None:
        data = get_feats()
        # h5.create_dataset(cache_file, data=data, compression='gzip')
        # else:
        # data = group[()]
        h5.close()
        return data
    else:
        return get_feats()


def extract(data: pd.DataFrame, features, **kwargs):
    """Function for extracting log-mel filterbank spectrogram features.

    Args:
        data: A pd.DataFrame containing datatset information and signals -- see docs for prep_data().
        features: "mfcc/lfcc"

    Returns:
        A pd.DataFrame containing features and metadata.
    """

    print("--------------- Extracting features ---------------")
    data = data.copy()
    data["features"] = data.apply(
        lambda x: extract_features(x, features, **kwargs), axis=1
    )

    data = data.drop(["signal"], axis=1)
    data = data.dropna().reset_index(drop=True)
    return data


def normalize(data: pd.DataFrame):
    """Function for normalizing features using z-score normalization.

    Args:
        data: A pd.DataFrame containing datatset information and features generated by extract().

    Returns:
        A pd.DataFrame containing normalized features and metadata.
    """

    data = data.copy()
    mean_std = data["features"].groupby(data["recording-id"]).apply(_get_mean_std)
    mean_std = mean_std.reset_index()
    if "level_1" in mean_std.columns:
        mean_std = mean_std.drop(["level_1"], axis=1)
    else:
        mean_std = mean_std.drop(["index"], axis=1)
    if "recording-id" in mean_std.columns:
        data = data.merge(mean_std, on="recording-id")
    else:
        data = pd.concat([data, mean_std], axis=1)
    print("--------------- Normalizing features --------------")
    data["normalized-features"] = data.apply(_calculate_norm, axis=1)
    data = data.drop(["features", "mean", "std"], axis=1)
    return data


def _calculate_norm(row: pd.DataFrame):
    """Auxiliary function used by normalize(). Calculates the normalized features from a row of
    a pd.DataFrame containing features and mean and standard deviation information (as generated
    by _get_mean_std()).

    Args:
        row: A row of a pd.DataFrame created by extract, with additional mean and standard deviation
        columns created by  _get_mean_std().

    Returns:
        An np.ndarray containing normalized features.
    """

    return np.array([(i - row["mean"]) / row["std"] for i in row["features"]])


def _get_mean_std(group: pd.core.groupby):
    """Auxiliary function used by normalize(). Calculates mean and standard deviation of a
    group of features.

    Args:
        group: A pd.GroupBy object referencing the features of a single wavefile (could be
        from multiple utterances).

    Returns:
        A pd.DataFrame with the mean and standard deviation of the group of features.
    """

    return pd.DataFrame(
        {
            "mean": [np.mean(np.vstack(group.to_numpy()))],
            "std": [np.std(np.vstack(group.to_numpy()))],
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="extract_feats.py", description="Extract log-mel spectrogram features."
    )

    parser.add_argument(
        "data_dir",
        type=str,
        help="a path to a Kaldi-style data directory containting 'wav.scp', and optionally 'segments'",
    )

    parser.add_argument(
        "out_dir",
        type=str,
        help="a path to an output directory where features and metadata will be saved as feats.h5",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="cnn_config.yaml",
        help="a path to a yaml config file for feature extraction",
    )

    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    # load config
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    print(config)
    data = extract_signal(args.data_dir)
    save(data, f"{args.out_dir}/signals.h5")

    data = extract(
        data,
        "lfcc",
        **config["lfcc"]
    )
    # data = normalize(data)

    save(data, f"{args.out_dir}/feats.h5")
