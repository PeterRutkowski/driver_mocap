import re
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as scipy_rotation


class BiwiParser:
    @classmethod
    def parse(cls, path_):
        """
        Parse files with annotations. Returns pd.Dataframe object.

        Parameters:
            path_: this parameter should end with "hpdp".
        """
        df = pd.DataFrame().assign(**cls._get_paths(path_))
        df = df.apply(cls._get_coordinates, axis=1)  # ~5 minutes
        df = df[df['rot'].apply(lambda x: type(x)) == list]  # filter out 2 invalid rows
        df = df.apply(cls._get_readable_coordinates, axis=1)
        return df

    @staticmethod
    def _get_paths(root_path):
        """
        Gets path of all annotations, images. The mothed created unique IDs as well. The ID consists of directory number
        and file number.
        """
        annotations_paths = list(root_path.glob('**/*.txt'))[1:]  # 'readme.txt' excluded
        image_paths = [path.with_name(path.stem[:-4] + 'rgb').with_suffix('.png') for path in annotations_paths]
        ids = [path.parent.stem + path.stem[5:-5] for path in annotations_paths]
        return {'id': ids, 'annotation_path': annotations_paths, 'image_path': image_paths}

    @staticmethod
    def _get_coordinates(row: pd.DataFrame):
        """
        Coordinates consist of rotation (3x3 matrix) and position (x, y, z).
        """
        with open(row['annotation_path'], 'r') as infile:
            text = infile.read()
        coords = [float(elem) for elem in re.findall("\-?\d+\.?\d+", text)]
        try:
            row['rot'], row['X'], row['Y'], row['Z'] = coords[:9], coords[9], coords[10], coords[11]
        except IndexError:
            pass  # ~2 annotations are invalid
        return row

    @staticmethod
    def _get_readable_coordinates(row: pd.DataFrame):
        """
        Converts 3x3 rotation matrix to (yaw, pitch, role) coordinates.
        """
        R = np.array(row['rot']).reshape(3, 3)
        R = np.transpose(R)
        row['pitch'], row['yaw'], row['roll'] = scipy_rotation.from_matrix(R).as_euler('xyz', degrees=True) * [1, -1, -1]
        return row
