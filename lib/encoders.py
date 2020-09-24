import numpy as np

class OrdinalEncoder:

    def __init__(self, first_index: int = 0, handle_unknown="max"):
        """
        :param first_index: classの最初の値
        :param handle_unknown: "value"=-1, max=class数+1, "zero"=0 で未知値を埋める
        """
        self.table_ = None
        self.first_index = first_index
        self.max_value = None
        self.handle_unknown = handle_unknown

    def fit(self, y_1d: list):
        """
        :param y_1d: ユニークであること
        :return: self
        """

        y_1d = np.asarray(y_1d)
        y_1d = np.ravel(y_1d)
        self.table_ = {val: i + self.first_index for i, val in enumerate(y_1d)}
        self.max_value = max(self.table_.values()) + 1

        return self

    def transform(self, y_1d) -> np.array:
        """
        :param y_1d: 1次元
        :return:
        """
        y_1d = np.asarray(y_1d)
        y_1d = np.ravel(y_1d)

        if self.handle_unknown == "value":
            # FIXME first_indexの値によって変える？もしくは引数にis_sequenceを入れる
            unknown_value = -1
        elif self.handle_unknown == "max":
            unknown_value = self.max_value
        elif self.handle_unknown == "zero":
            unknown_value = 0

        if len(y_1d) == 0:
            return np.array([])
        else:
            return np.array([self.table_[v] if v in self.table_ else unknown_value for v in y_1d])

    def get_class_num(self) -> int:
        """
        class数を得る
        :return:
        """
        return self.max_value - self.first_index

    def get_max_value(self) -> int:
        """
        classのmax+1（＝不明値）を得る
        :return:
        """
        return self.max_value

    def restore_from_dict(self, dict, first_index=None):
        """
        val: indexのdictからencoderを復元する
        :param dict:
        :param first_index:
        :return:
        """
        self.table_ = dict
        self.max_value = max(self.table_.values()) + 1
        if not first_index:
            self.first_index = min(dict.values())

        return self
