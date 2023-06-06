
from torch.utils.data import Dataset

def read_data():
    # Read data
    # Read data
    f1 = open("/content/gdrive/My Drive/toot-sc-bert/datasets/transporter_substrate_class/Transpoter_substrates.fasta",
              'r')
    all_data = f1.readlines()

    f2 = open("/content/gdrive/My Drive/toot-sc-bert/datasets/transporter_substrate_class/substrate_classes_all.csv",
              'r')
    next(f2)
    all_tags = f2.readlines()

    f3 = open("/content/gdrive/My Drive/toot-sc-bert/datasets/transporter_substrate_class/substrate_classes_indep.csv",
              'r')
    next(f3)
    test_indep = f3.readlines()
    id_seq = {}
    id_class = {}
    id_test = {}
    id_train = []

    for i in range(0, len(all_data)):
        if all_data[i].startswith('>') and all_data[i] not in id_seq.keys():
            id_seq[all_data[i].strip('\n').strip('>')] = all_data[i + 1]

    for line in all_tags:
        line = line.split(',')
        if line[0] not in id_class.keys():
            id_class[line[0]] = int(line[1]) - 1

    for line in test_indep:
        line = line.split(',')
        if line[0] not in id_test.keys():
            id_test[line[0]] = int(line[1]) - 1

    id_train = id_seq.keys() - id_test.keys()
    test_set = []
    for id in id_test.keys():
        id_seq[id].replace('U', 'X')
        id_seq[id].replace('Z', 'X')
        id_seq[id].replace('O', 'X')
        id_seq[id].replace('B', 'X')
        test_set.append((' '.join(id_seq[id]), id_class[id]))

    train_set = []
    for id in id_train:
        id_seq[id].replace('U', 'X')
        id_seq[id].replace('Z', 'X')
        id_seq[id].replace('O', 'X')
        id_seq[id].replace('B', 'X')
        train_set.append((' '.join(id_seq[id]), id_class[id]))

    return train_set, test_set



class sc_dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

