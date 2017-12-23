import logging

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(name)s: %(message)s (%(asctime)s; %(filename)s:%(lineno)d)',
                    datefmt="%Y-%m-%d %H:%M:%S", )
logger = logging.getLogger(__file__)

FOLDS = [[100290, 100260, 100070, 100440, 100355, 100245, 100281],
         [100006, 100001, 100326, 100451, 100076, 100416, 100151],
         [100182, 100137, 100287, 100142, 100437, 100317, 100313],
         [100028, 100013, 100428, 100418, 100093, 100463, 100208],
         [100154, 100369, 100249, 100459, 100274, 100394]]

