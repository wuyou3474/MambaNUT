import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class NAT2024Dataset(BaseDataset):

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.nat2024_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                                                                           sequence_path=sequence_path, frame=frame_num,
                                                                           nz=nz, ext=ext) for frame_num in
                  range(start_frame + init_omit, end_frame + 1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64, backend='numpy')

        return Sequence(sequence_info['name'], frames, 'uav', ground_truth_rect[init_omit:, :],
                        object_class=sequence_info['object_class'])

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {'name': 'L05021', 'path': 'data_seq/L05021', 'startFrame': 1, 'endFrame': 1701, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L05021.txt', 'object_class': 'other'},
             {'name': 'L10002', 'path': 'data_seq/L10002', 'startFrame': 1, 'endFrame': 1946, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L10002.txt', 'object_class': 'other'},
             {'name': 'L09002', 'path': 'data_seq/L09002', 'startFrame': 1, 'endFrame': 1950, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L09002.txt', 'object_class': 'other'},
             {'name': 'L04003', 'path': 'data_seq/L04003', 'startFrame': 1, 'endFrame': 1505, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L04003.txt', 'object_class': 'other'},
             {'name': 'L05010', 'path': 'data_seq/L05010', 'startFrame': 1, 'endFrame': 1566, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L05010.txt', 'object_class': 'other'},
             {'name': 'L05014', 'path': 'data_seq/L05014', 'startFrame': 1, 'endFrame': 1691, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L05014.txt', 'object_class': 'other'},
             {'name': 'L05020', 'path': 'data_seq/L05020', 'startFrame': 1, 'endFrame': 1506, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L05020.txt', 'object_class': 'other'},
             {'name': 'L01004', 'path': 'data_seq/L01004', 'startFrame': 1, 'endFrame': 1551, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L01004.txt', 'object_class': 'other'},
             {'name': 'L10003', 'path': 'data_seq/L10003', 'startFrame': 1, 'endFrame': 1614, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L10003.txt', 'object_class': 'other'},
             {'name': 'L05011', 'path': 'data_seq/L05011', 'startFrame': 1, 'endFrame': 2008, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L05011.txt', 'object_class': 'other'},
             {'name': 'L10001', 'path': 'data_seq/L10001', 'startFrame': 1, 'endFrame': 1755, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L10001.txt', 'object_class': 'other'},
             {'name': 'L01005', 'path': 'data_seq/L01005', 'startFrame': 1, 'endFrame': 1612, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L01005.txt', 'object_class': 'other'},
             {'name': 'L03001', 'path': 'data_seq/L03001', 'startFrame': 1, 'endFrame': 2067, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L03001.txt', 'object_class': 'other'},
             {'name': 'L01002', 'path': 'data_seq/L01002', 'startFrame': 1, 'endFrame': 1869, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L01002.txt', 'object_class': 'other'},
             {'name': 'L04001', 'path': 'data_seq/L04001', 'startFrame': 1, 'endFrame': 2160, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L04001.txt', 'object_class': 'other'},
             {'name': 'L05012', 'path': 'data_seq/L05012', 'startFrame': 1, 'endFrame': 1537, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L05012.txt', 'object_class': 'other'},
             {'name': 'L06001', 'path': 'data_seq/L06001', 'startFrame': 1, 'endFrame': 1692, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L06001.txt', 'object_class': 'other'},
             {'name': 'L05019', 'path': 'data_seq/L05019', 'startFrame': 1, 'endFrame': 1553, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L05019.txt', 'object_class': 'other'},
             {'name': 'L05002', 'path': 'data_seq/L05002', 'startFrame': 1, 'endFrame': 1512, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L05002.txt', 'object_class': 'other'},
             {'name': 'L05006', 'path': 'data_seq/L05006', 'startFrame': 1, 'endFrame': 1530, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L05006.txt', 'object_class': 'other'},
             {'name': 'L05016', 'path': 'data_seq/L05016', 'startFrame': 1, 'endFrame': 1778, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L05016.txt', 'object_class': 'other'},
             {'name': 'L05004', 'path': 'data_seq/L05004', 'startFrame': 1, 'endFrame': 1530, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L05004.txt', 'object_class': 'other'},
             {'name': 'L05017', 'path': 'data_seq/L05017', 'startFrame': 1, 'endFrame': 1591, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L05017.txt', 'object_class': 'other'},
             {'name': 'L01003', 'path': 'data_seq/L01003', 'startFrame': 1, 'endFrame': 1626, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L01003.txt', 'object_class': 'other'},
             {'name': 'L01001', 'path': 'data_seq/L01001', 'startFrame': 1, 'endFrame': 1585, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L01001.txt', 'object_class': 'other'},
             {'name': 'L05013', 'path': 'data_seq/L05013', 'startFrame': 1, 'endFrame': 2238, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L05013.txt', 'object_class': 'other'},
             {'name': 'L08001', 'path': 'data_seq/L08001', 'startFrame': 1, 'endFrame': 2258, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L08001.txt', 'object_class': 'other'},
             {'name': 'L05007', 'path': 'data_seq/L05007', 'startFrame': 1, 'endFrame': 1503, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L05007.txt', 'object_class': 'other'},
             {'name': 'L05005', 'path': 'data_seq/L05005', 'startFrame': 1, 'endFrame': 1501, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L05005.txt', 'object_class': 'other'},
             {'name': 'L02001', 'path': 'data_seq/L02001', 'startFrame': 1, 'endFrame': 1649, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L02001.txt', 'object_class': 'other'},
             {'name': 'L03002', 'path': 'data_seq/L03002', 'startFrame': 1, 'endFrame': 1967, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L03002.txt', 'object_class': 'other'},
             {'name': 'L04002', 'path': 'data_seq/L04002', 'startFrame': 1, 'endFrame': 2066, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L04002.txt', 'object_class': 'other'},
             {'name': 'L05008', 'path': 'data_seq/L05008', 'startFrame': 1, 'endFrame': 1643, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L05008.txt', 'object_class': 'other'},
             {'name': 'L09001', 'path': 'data_seq/L09001', 'startFrame': 1, 'endFrame': 1571, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L09001.txt', 'object_class': 'other'},
             {'name': 'L05003', 'path': 'data_seq/L05003', 'startFrame': 1, 'endFrame': 1719, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L05003.txt', 'object_class': 'other'},
             {'name': 'L04004', 'path': 'data_seq/L04004', 'startFrame': 1, 'endFrame': 1523, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L04004.txt', 'object_class': 'other'},
             {'name': 'L05001', 'path': 'data_seq/L05001', 'startFrame': 1, 'endFrame': 2856, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L05001.txt', 'object_class': 'other'},
             {'name': 'L07001', 'path': 'data_seq/L07001', 'startFrame': 1, 'endFrame': 1677, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L07001.txt', 'object_class': 'other'},
             {'name': 'L06002', 'path': 'data_seq/L06002', 'startFrame': 1, 'endFrame': 1487, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L06002.txt', 'object_class': 'other'},
             {'name': 'L05015', 'path': 'data_seq/L05015', 'startFrame': 1, 'endFrame': 2001, 'nz': 6, 'ext': 'jpg',
              'anno_path': 'anno/L05015.txt', 'object_class': 'other'}
        ]

        return sequence_info_list