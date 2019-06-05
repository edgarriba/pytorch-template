import os


def create_directory(dir_path):
    """Creates an empty directory.
    Args:
        dir_path (str): the absolute path to the directory to create.
    """
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)