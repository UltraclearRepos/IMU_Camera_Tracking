"""Fill Dobot timestamps below and run this file to create batch configs."""

import csv
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = MODULE_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "Data"
BASE_CONFIG_DIR = MODULE_DIR / "batch_configs"
OUTPUT_DIR = MODULE_DIR / "batch_configs_from_timestamps"
VIDEO_FPS = 30.0

# For every recording enter:
# (mapping_start_sync_timestamp, mapping_end_sync_timestamp,
#  tracking_start_sync_timestamp)
TIMESTAMP_RANGES = {
    # "Line": {
    #     "close-dark-nolight_Speed-3_2026-07-28_17.16.20": (1785251782.289967, 1785251794.685367, 1785251794.741467),
    #     "close-dark-withlight_Speed-3_2026-07-28_17.17.50": (1785251871.595167, 1785251884.663167, 1785251884.718267),
    #     "close-white-nolight_Speed-3_2026-07-28_17.14.02": (1785251644.120367, 1785251657.212667, 1785251657.285567),
    #     "close-white-withlight_Speed-3_2026-07-28_17.12.37": (1785251559.303867, 1785251572.202167, 1785251572.258967),
    #     "far-dark-nolight_Speed-3_2026-07-28_17.04.19": (1785251061.673367, 1785251074.407267, 1785251074.460467),
    #     "far-dark-withlight_Speed-3_2026-07-28_17.02.52": (1785250973.618367, 1785250986.898567, 1785250986.954967),
    #     "far-white-nolight_Speed-3_2026-07-28_17.06.45": (1785251207.324367, 1785251220.051967, 1785251220.116367),
    #     "far-white-withlight_Speed-3_2026-07-28_17.08.22": (1785251303.553567, 1785251316.853167, 1785251316.913367),
    #     "initialpos-dark-nolight_Speed-3_2026-07-28_16.55.02": (1785250504.351767, 1785250517.123667, 1785250517.170467),
    #     "initialpos-dark-withlight_Speed-3_2026-07-28_16.57.56": (1785250677.767667, 1785250690.806467, 1785250690.863167),
    #     "initialpos-white-nolight_Speed-3_2026-07-29_17.47.53": (1785340075.128067, 1785340087.984367, 1785340088.048867),
    #     "initialpos-white-withlight_Speed-3_2026-07-29_17.46.25": (1785339986.974667, 1785340000.150867, 1785340000.205667),
    #     "initialpos_Speed-3_2026-07-28_16.38.48": (1785249529.241667, 1785249542.460267, 1785249542.517167),
    # },
    # "LineArc-1-2cm": {
    #     "arc1cm-close-dark-nolight_Speed-3_2026-07-29_16.12.21": (1785334343.129467, 1785334355.208167, 1785334355.255567),
    #     "arc1cm-close-dark-withlight_Speed-3_2026-07-29_16.08.30": (1785334111.973467, 1785334124.133167, 1785334124.201067),
    #     "arc1cm-close-white-nolight_Speed-3_2026-07-29_16.29.21": (1785335362.592567, 1785335374.628167, 1785335374.684067),
    #     "arc1cm-close-white-withlight_Speed-3_2026-07-29_16.32.22": (1785335544.119967, 1785335556.032267, 1785335556.112667),
    #     "arc1cm-far-dark-nolight_Speed-3_2026-07-29_16.52.01": (1785336723.340967,1785336735.259667, 1785336735.312767),
    #     "arc1cm-far-dark-withlight_Speed-3_2026-07-29_16.53.23": (1785336804.590967, 1785336816.623267, 1785336816.701167),
    #     "arc1cm-far-white-nolight_Speed-3_2026-07-29_17.03.56": (1785337438.505967, 1785337450.466167, 1785337450.534267),
    #     "arc1cm-far-white-withlight_Speed-3_2026-07-29_17.02.26": (1785337347.603467, 1785337359.712467, 1785337359.763667),
    #     "arc1cm-initial-dark-nolight_Speed-3_2026-07-29_16.48.44": (1785336525.952367, 1785336537.881067, 1785336537.922867),
    #     "arc1cm-initial-dark-withlight_Speed-3_2026-07-29_16.47.01": (1785336422.922667, 1785336434.989967, 1785336435.049167),
    #     "arc1cm-initial-white-nolight_Speed-3_2026-07-29_16.35.50": (1785335752.160767, 1785335764.320567, 1785335764.367367),
    #     "arc1cm-initial-white-withlight_Speed-3_2026-07-29_16.34.29": (1785335671.053767, 1785335683.214367, 1785335683.261567),
    #     "arc2cm-close-dark-nolight_Speed-3_2026-07-29_16.14.42": (1785334483.573867, 1785334495.978067, 1785334496.029467),
    #     "arc2cm-close-dark-withlight_Speed-3_2026-07-29_16.15.59": (1785334560.972467, 1785334573.036767, 1785334573.080467),
    #     "arc2cm-close-white-nolight_Speed-3_2026-07-29_16.27.38": (1785335259.664467, 1785335271.939167, 1785335272.000667),
    #     "arc2cm-close-white-withlight_Speed-3_2026-07-29_16.25.40": (1785335142.198867, 1785335154.483767, 1785335154.524267),
    #     "arc2cm-far-dark-nolight_Speed-3_2026-07-29_16.56.25": (1785336987.094967, 1785336999.326467, 1785336999.399467),
    #     "arc2cm-far-dark-withlight_Speed-3_2026-07-29_16.55.05": (1785336907.221667, 1785336919.285667, 1785336919.350667,),
    #     "arc2cm-far-white-nolight_Speed-3_2026-07-29_16.59.01": (1785337142.727167, 1785337154.712167, 1785337154.764067),
    #     "arc2cm-far-white-withlight_Speed-3_2026-07-29_17.00.49": (1785337251.821667, 1785337263.645467, 1785337263.702367),
    #     "arc2cm-initial-dark-nolight_Speed-3_2026-07-29_16.44.19": (1785336261.277467, 1785336273.521567, 1785336273.573967),
    #     "arc2cm-initial-dark-withlight_Speed-3_2026-07-29_16.45.45": (1785336347.012167, 1785336359.220267, 1785336359.264367),
    #     "arc2cm-initial-white-nolight_Speed-3_2026-07-29_16.37.23": (1785335845.408667, 1785335857.645867, 1785335857.701467),
    #     "arc2cm-initial-white-withlight_Speed-3_2026-07-29_16.38.41": (1785335922.701067, 1785335934.810067, 1785335934.884067),
    # },
    # "LineWithR": {
    #     "close-black-nolight-25deg_Speed-3_2026-07-30_13.38.42": (1785411523.849567, 1785411537.358067, 1785411537.428167),
    #     "close-black-withlight-25deg_Speed-3_2026-07-30_13.37.05": (1785411426.638967, 1785411440.072067, 1785411440.129367),
    #     "close-white-nolight-25deg_Speed-3_2026-07-30_13.31.23": (1785411084.737467, 1785411097.948967, 1785411098.009067),
    #     "close-white-withlight-25deg_Speed-3_2026-07-30_13.33.14": (1785411195.554567, 1785411208.686967, 1785411208.762967),
    #     "far-black-nolight-25deg_Speed-3_2026-07-30_13.41.36": (1785411697.352767, 1785411710.609667, 1785411710.648567),
    #     "far-black-withlight-25deg_Speed-3_2026-07-30_13.44.45": (1785411886.768867, 1785411899.880167, 1785411899.936867),
    #     "far-white-nolight-25deg_Speed-3_2026-07-30_13.20.24": (1785410426.053767, 1785410439.183467, 1785410439.229267),
    #     "far-white-withlight-25deg_Speed-3_2026-07-30_13.21.53": (1785410515.072767, 1785410528.399167, 1785410528.456167),
    #     "initial-black-nolight-25deg_Speed-3_2026-07-30_13.50.30": (1785412231.471567, 1785412244.645467, 1785412244.677967),
    #     "initial-black-withlight-25deg_Speed-3_2026-07-30_13.46.33": (1785411995.215567, 1785412008.396767, 1785412008.432267),
    #     "initial-white-nolight-25deg_Speed-3_2026-07-30_13.28.57": (1785410938.788167, 1785410951.895467, 1785410951.951967),
    #     "initial-white-withlight-25deg_Speed-3_2026-07-30_13.27.38": (1785410859.613867, 1785410872.675767, 1785410872.725067),
    # },
    # "OnlyR": {
    #     "close-black-nolight-25deg_Speed-3_2026-07-30_14.08.24": (1785413306.245367, 1785413314.902467, 1785413314.960567),
    #     "close-black-withlight-25deg_Speed-3_2026-07-30_14.09.41": (1785413383.194667, 1785413392.049667, 1785413392.098767),
    #     "close-white-nolight-25deg_Speed-3_2026-07-30_13.10.31": (1785409832.793767, 1785409841.576167, 1785409841.661067),
    #     "close-white-withlight-25deg_Speed-3_2026-07-30_13.11.32": (1785409893.960067, 1785409902.673167, 1785409902.717667),
    #     "far-black-nolight-25deg_Speed-3_2026-07-30_14.00.25": (1785412827.192967, 1785412835.733367, 1785412835.810267),
    #     "far-black-withlight-25deg_Speed-3_2026-07-30_13.58.22": (1785412704.008567, 1785412712.596267, 1785412712.653267),
    #     "far-white-nolight-25deg_Speed-3_2026-07-30_13.14.23": (1785410065.314367, 1785410074.093767, 1785410074.149367),
    #     "far-white-withlight-25deg_Speed-3_2026-07-30_13.13.13": (1785409994.426567, 1785410003.492567, 1785410003.558267),
    #     "initial-black-nolight-25deg_Speed-3_2026-07-30_13.55.04": (1785412505.521367, 1785412514.233267, 1785412514.301067),
    #     "initial-black-withlight-25deg_Speed-3_2026-07-30_13.56.04": (1785412565.980167, 1785412574.768867, 1785412574.835967),
    #     "initial-white-nolight-25deg_Speed-3_2026-07-30_13.07.33": (1785409654.882067, 1785409663.838767, 1785409663.889767),
    #     "initial-white-withlight-25deg_Speed-3_2026-07-30_13.06.03": (1785409565.300167, 1785409574.063667, 1785409574.143867),
    # },
    # "Cylinder": {
    #     "close_25mm_Arc180-Speed-3_2026-08-18_17.52.55": (1787068375.603167, 1787068403.501567, 1787068403.552967),
    #     "close_25mm_Arc180-Speed-3_2026-08-18_17.53.39": (1787068419.826467, 1787068447.593067, 1787068447.644267),
    #     "close_25mm_Arc180-Speed-3_2026-08-18_17.54.22": (1787068462.517667, 1787068490.268367, 1787068490.314867),
    #     "close_25mm_Arc180-Speed-3_2026-08-18_17.55.10": (1787068511.063667, 1787068538.726467, 1787068538.801067),
    #     "close_25mm_Arc180-Speed-3_2026-08-18_17.55.57": (1787068557.772567, 1787068585.613367, 1787068585.678067),
    #     "close_25mm_Arc180-Speed-3_2026-08-19_13.57.03": (1787140623.816567, 1787140650.755467, 1787140650.812367),
    #     "close_25mm_Arc180-Speed-3_2026-08-19_13.57.47": (1787140667.970867, 1787140694.996667, 1787140695.061167),
    #     "close_25mm_Arc180-Speed-3_2026-08-19_14.00.31": (1787140831.655967, 1787140858.607667, 1787140858.652967),
    #     "close_25mm_Arc180-Speed-3_2026-08-19_14.01.37": (1787140898.302667, 1787140925.178367, 1787140925.233967),
    #     "close_25mm_Arc180-Speed-3_2026-08-19_14.03.54": (1787141034.709567, 1787141061.681167, 1787141061.749667),
    #     "initial_50mm_Arc180-Speed-3_2026-08-18_17.47.36": (1787068057.181767, 1787068090.074867, 1787068090.125767),
    #     "initial_50mm_Arc180-Speed-3_2026-08-18_17.48.29": (1787068110.255867, 1787068143.127467, 1787068143.196267),
    #     "initial_50mm_Arc180-Speed-3_2026-08-18_17.49.22": (1787068162.934567, 1787068195.866867, 1787068195.934467),
    #     "initial_50mm_Arc180-Speed-3_2026-08-18_17.50.15": (1787068215.472767, 1787068248.364967, 1787068248.439867),
    #     "initial_50mm_Arc180-Speed-3_2026-08-18_17.51.13": (1787068273.375067, 1787068306.407767, 1787068306.470967),
    #     "initial_50mm_Arc180-Speed-3_2026-08-19_14.15.35": (1787141735.995867, 1787141768.290967, 1787141768.346867),
    #     "initial_50mm_Arc180-Speed-3_2026-08-19_14.16.20": (1787141780.873267, 1787141812.988867, 1787141813.057667),
    #     "initial_50mm_Arc180-Speed-3_2026-08-19_14.17.06": (1787141827.201467, 1787141859.454667, 1787141859.518067),
    #     "initial_50mm_Arc180-Speed-3_2026-08-19_14.17.50": (1787141871.215667, 1787141903.416567, 1787141903.483267),
    #     "initial_50mm_Arc180-Speed-3_2026-08-19_14.18.45": (1787141925.425967, 1787141957.648967, 1787141957.700367),
    # },
    "CylinderRepetitions": {
        "close_25mm_Arc180-Speed-3_2026-08-20_14.42.34": (1787229754.838067, 1787229785.528267, 1787229785.595967),
        "close_25mm_Arc180-Speed-3_2026-08-20_15.33.48": (1787232828.832667, 1787232859.544667, 1787232859.594067),
        "initial_50mm_Arc180-Speed-3_2026-08-20_14.39.08": (1787229548.399767, 1787229584.401267, 1787229584.469567),
        "initial_50mm_Arc180-Speed-3_2026-08-20_15.30.28": (1787232629.088867, 1787232665.031767, 1787232665.096267)
    }
}


def get_video_start_timestamp(data_folder, recording_name):
    path = DATA_DIR / data_folder / "video_timestamps" / f"{recording_name}.csv"
    with path.open(newline="", encoding="utf-8") as file:
        return float(next(csv.DictReader(file))["start_timestamp"])


def timestamp_to_frame(timestamp, video_start_timestamp):
    return max(round((timestamp - video_start_timestamp) * VIDEO_FPS), 1)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    for data_folder, recordings in TIMESTAMP_RANGES.items():
        config_path = BASE_CONFIG_DIR / f"{data_folder}.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for recording_name, timestamps in recordings.items():
            video_start = get_video_start_timestamp(data_folder, recording_name)
            start, end, tracking = timestamps
            mapping_start_frame = timestamp_to_frame(start, video_start)
            mapping_end_frame = timestamp_to_frame(end, video_start)
            tracking_start_frame = timestamp_to_frame(tracking, video_start)

            if tracking_start_frame == mapping_end_frame:
                tracking_start_frame += 1

            config["recordings"][recording_name].update(
                mapping_start_frame=mapping_start_frame,
                mapping_end_frame=mapping_end_frame,
                tracking_start_frame=tracking_start_frame,
            )
        output_path = OUTPUT_DIR / f"{data_folder}.json"
        output_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
