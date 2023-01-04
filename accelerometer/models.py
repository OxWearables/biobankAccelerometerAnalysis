import pathlib

ROOT_DIR = pathlib.Path(__file__).parent
MODEL_VER = "10Feb2022"
MODEL_DIR = ROOT_DIR / "models"
MODEL_URL = "https://wearables-files.ndph.ox.ac.uk/files/models"


MODELS = {
    'willetts': {
        "pth": MODEL_DIR / MODEL_VER / "willetts/model.tar",
        "url": f"{MODEL_URL}/{MODEL_VER}/willetts/model.tar",
    },
    'doherty': {
        "pth": MODEL_DIR / MODEL_VER / "doherty/model.tar",
        "url": f"{MODEL_URL}/{MODEL_VER}/doherty/model.tar",
    },
    'walmsley': {
        "pth": MODEL_DIR / MODEL_VER / "walmsley/model.tar",
        "url": f"{MODEL_URL}/{MODEL_VER}/walmsley/model.tar",
    },
    'chan': {
        "pth": MODEL_DIR / "01Jan2023" / "chan/model.tar",
        "url": "https://tinyurl.com/yuzn99zr",
    },

    'willetts-10Feb2022': {
        "pth": MODEL_DIR / "10Feb2022" / "willetts/model.tar",
        "url": f"{MODEL_URL}/10Feb2022/willetts/model.tar",
    },
    'doherty-10Feb2022': {
        "pth": MODEL_DIR / "10Feb2022" / "doherty/model.tar",
        "url": f"{MODEL_URL}/10Feb2022/doherty/model.tar",
    },
    'walmsley-10Feb2022': {
        "pth": MODEL_DIR / "10Feb2022" / "walmsley/model.tar",
        "url": f"{MODEL_URL}/10Feb2022/walmsley/model.tar",
    },
    'chan-01Jan2023': {
        "pth": MODEL_DIR / "01Jan2023" / "chan/model.tar",
        "url": "https://tinyurl.com/yuzn99zr",
    }

}
