import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Visualizer:
    chosen_numbers = [3, 31, 107, 129, 144, 220, 291, 321, 342, 366, 418, 435, 447, 500]

    @classmethod
    def plot_feature(cls, df_, feature):
        """
        Shows what each feature represents. Thanks to this you do need to remember e.q. what "yaw" is.
        Parameters:
            feature: this parameters must be within {"yaw", "pitch", "role"} set.
        """
        id_list = ['01_' + str(n).zfill(5) for n in cls.chosen_numbers]
        rows = df_[df_['id'].isin(id_list)].sort_values(by=[feature]).reset_index()

        f, axarr = plt.subplots(4, 4)
        for index, row in rows.iterrows():
            image_path = row['image_path']
            axarr[divmod(index, 4)].imshow(mpimg.imread(image_path))
            axarr[divmod(index, 4)].set_ylabel(int(row[feature]))

        return rows
