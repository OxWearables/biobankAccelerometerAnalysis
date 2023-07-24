import pathlib

MODEL_DIR = pathlib.Path(__file__).parent / "models"
MODEL_ROOT_URL = "https://wearables-files.ndph.ox.ac.uk/files/models/biobankAccelerometerAnalysis"


MODELS = {

    'willetts': {
        "pth": MODEL_DIR / "willetts" / "20220210_sk_1_0_2.tar",
        "url": f"{MODEL_ROOT_URL}/willetts/20220210_sk_1_0_2.tar",
    },

    'doherty': {
        "pth": MODEL_DIR / "doherty" / "20220210_sk_1_0_2.tar",
        "url": f"{MODEL_ROOT_URL}/doherty/20220210_sk_1_0_2.tar",
    },

    'walmsley': {
        "pth": MODEL_DIR / "walmsley" / "20220210_sk_1_0_2.tar",
        "url": f"{MODEL_ROOT_URL}/walmsley/20220210_sk_1_0_2.tar",
    },

    'chan': {
        "pth": MODEL_DIR / "chan" / "20230103_sk_1_0_2.tar",
        "url": f"{MODEL_ROOT_URL}/chan/20230103_sk_1_0_2.tar",
    },

    'chanw': {
        "pth": MODEL_DIR / "chanw" / "20230106_sk_1_0_2.tar",
        "url": f"{MODEL_ROOT_URL}/chanw/20230106_sk_1_0_2.tar"
    },

}
