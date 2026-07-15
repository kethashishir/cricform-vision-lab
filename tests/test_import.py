from cricform import __version__
from cricform.config import project_paths


def test_package_imports() -> None:
    assert __version__ == "0.1.0"


def test_project_paths_exist() -> None:
    paths = project_paths()

    assert paths["project_root"].exists()
    assert paths["data_dir"].exists()
    assert paths["raw_video_dir"].exists()
    assert paths["interim_dir"].exists()
    assert paths["processed_dir"].exists()
    assert paths["outputs_dir"].exists()
