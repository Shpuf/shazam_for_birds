import os
from collections import defaultdict
import numpy as np
from maad import sound, rois, features
from maad.util import power2dB, format_features
from pathlib import Path
from DatabaseMaker import chdir

FEATURES = ['centroid_f', 'centroid_t', 'duration_t', 'bandwidth_f', 'area_tf']

N_INPUTS = 250

N_FEATURES = len(FEATURES)

INPUTS = np.empty(shape=(N_INPUTS, N_FEATURES), dtype=float)
LABELS = []


def main():
    def label_by_filename(filename: str) -> str:
        return filename.split('_')[0]

    label_count = defaultdict(int)

    global INPUTS
    global LABELS
    with chdir('audio_data'):
        index = 0
        for filename in os.listdir(Path().absolute()):
            try:
                label = label_by_filename(filename)
                if sum(label_count.values()) >= N_INPUTS:
                    break
                elif label_count[label] >= N_INPUTS // N_FEATURES:
                    continue

                s, fs = sound.load(filename)

                dB_max = 70

                Sxx_power, tn, fn, ext = sound.spectrogram(s, fs, nperseg=1024, noverlap=1024 // 2)

                Sxx_db = power2dB(Sxx_power) + dB_max

                Sxx_power_noNoise = sound.median_equalizer(Sxx_power,  **{'extent': ext})
                Sxx_db_noNoise = power2dB(Sxx_power_noNoise)

                Sxx_db_noNoise_smooth = sound.smooth(Sxx_db_noNoise, std=0.5,
                                                    savefig=None,
                                                     **{'vmin': 0, 'vmax': dB_max, 'extent': ext})

                im_mask = rois.create_mask(im=Sxx_db_noNoise_smooth, mode_bin='relative',
                                           bin_std=8, bin_per=0.5,
                                           verbose=False)

                im_rois, df_rois = rois.select_rois(im_mask, min_roi=25, max_roi=None,
                                                    **{'extent': ext})

                df_rois = format_features(df_rois, tn, fn)

                df_centroid = features.centroid_features(Sxx_db, df_rois, im_rois)

                df_centroid = format_features(df_centroid, tn, fn)

                n_centroids = len(df_centroid['area_xy'])

                for centroid in range(n_centroids):
                    if label_count[label] >= N_INPUTS // N_FEATURES:
                        break
                    INPUTS[index] = np.array([df_centroid[i][centroid] for i in FEATURES])
                    LABELS.append(label)
                    label_count[label] += 1
                    index += 1
            except IndexError:
                continue
    np.save('inputs.npy', INPUTS)
    np.save('labels.npy', np.array(LABELS))



if __name__ == '__main__':
    main()
