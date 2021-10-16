import pathlib

ROOT_DIR = pathlib.Path(__file__).parent
MODEL_DIR = ROOT_DIR / "activityModels/"
MODEL_DIR_URL = "http://gas.ndph.ox.ac.uk/aidend/accModels/"


MODELS = {
    'willetts': {
        "pth": MODEL_DIR / "willetts-jan21.tar",
        "url": MODEL_DIR_URL + "willetts-jan21.tar",
    },
    'doherty': {
        "pth": MODEL_DIR / "doherty-jan21.tar",
        "url": MODEL_DIR_URL + "doherty-jan21.tar",
    },
    'walmsley': {
        "pth": MODEL_DIR / "walmsley-jan21.tar",
        "url": MODEL_DIR_URL + "walmsley-jan21.tar",
    },
    'willetts-jan21': {
        "pth": MODEL_DIR / "willetts-jan21.tar",
        "url": MODEL_DIR_URL + "willetts-jan21.tar",
    },
    'doherty-jan21': {
        "pth": MODEL_DIR / "doherty-jan21.tar",
        "url": MODEL_DIR_URL + "doherty-jan21.tar",
    },
    'walmsley-jan21': {
        "pth": MODEL_DIR / "walmsley-jan21.tar",
        "url": MODEL_DIR_URL + "walmsley-jan21.tar",
    },
    'willetts-may20': {
        "pth": MODEL_DIR / "willetts-may20.tar",
        "url": MODEL_DIR_URL + "willetts-may20.tar",
    },
    'doherty-may20': {
        "pth": MODEL_DIR / "doherty-may20.tar",
        "url": MODEL_DIR_URL + "doherty-may20.tar",
    },
    'walmsley-nov20': {
        "pth": MODEL_DIR / "walmsley-nov20.tar",
        "url": MODEL_DIR_URL + "walmsley-nov20.tar",
    },
}
