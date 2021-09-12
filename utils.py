import time,math,random,torch,os
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids


# This method takes in a confusion matrix (array) and compute precision, recall and F1
# The metrics are descibed in Ning, et. al. EMNLP 2019
def confusion2prf(confusion):
    if confusion.shape[0] == 6:
        f1 = 1.0 * np.sum([confusion[i][i] for i in range(confusion.shape[0])]) / np.sum(confusion)
        prec = f1
        rec = f1
        return prec,rec,f1

    if np.sum([confusion[i][i] for i in range(confusion.shape[0]-1)]) == 0:
        return 0., 0., 0.
    prec = 1.0 * np.sum([confusion[i][i] for i in range(confusion.shape[0]-1)]) / (np.sum(confusion[:,:confusion.shape[0]-1]))
    rec = 1.0 * np.sum([confusion[i][i] for i in range(confusion.shape[0]-1)]) / (np.sum(confusion[:confusion.shape[0]-1,:]))
    f1 = 2.0 / (1.0 / prec + 1.0 / rec)
    return prec,rec,f1


def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return category_i


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed+1)
    torch.cuda.manual_seed_all(seed+2)
    np.random.seed(seed+3)
    os.environ['PYTHONHASHSEED'] = str(seed+4)
    torch.backends.cudnn.deterministic = True


# download ELMo model files
def load_elmo(option='small'):
    if option == 'small':
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
    elif option == 'medium':
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"
    elif option == 'original':
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    else:
        print("option (%s) is not specified. Using small as default" % option)
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
    elmo = Elmo(options_file, weight_file, 1, dropout=0)
    return elmo
    