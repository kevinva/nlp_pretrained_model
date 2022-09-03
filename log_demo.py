import logging
import time
import random

logger = logging.getLogger('hoho_logger')
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(f'./log/hoho_{int(time.time())}.log')
fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s')

fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

# logging.basicConfig(filename='./log.txt', level=logging.DEBUG, format='[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s')


for i in range(1000):
    time.sleep(random.randint(0, 3))
    logger.info(f'This is a log message. {i}')