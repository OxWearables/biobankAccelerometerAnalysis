import pathlib

MODEL_DIR = pathlib.Path(__file__).parent / "models"
MODEL_ROOT_URL = "https://wearables-files.ndph.ox.ac.uk/files/models/biobankAccelerometerAnalysis"


MODELS = {

    'willetts': {
        "pth": MODEL_DIR / "willetts" / "20220210.tar",
        "url": f"{MODEL_ROOT_URL}/willetts/20220210.tar",
    },

    'doherty': {
        "pth": MODEL_DIR / "doherty" / "20220210.tar",
        "url": f"{MODEL_ROOT_URL}/doherty/20220210.tar",
    },

    'walmsley': {
        "pth": MODEL_DIR / "walmsley" / "20220210.tar",
        "url": f"{MODEL_ROOT_URL}/walmsley/20220210.tar",
    },

    'chan': {
        "pth": MODEL_DIR / "chan" / "20230103.tar",
        "url": f"{MODEL_ROOT_URL}/chan/20230103.tar",

    },

    'chanw': {
        "pth": MODEL_DIR / "chanw" / "20230106.tar",
        "url": f"{MODEL_ROOT_URL}/chanw/20230106.tar"
    },

}
