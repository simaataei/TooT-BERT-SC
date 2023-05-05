

def read_data():
    # Read data
    f1 = open("./Datasets/Transpoter_substrates.fasta", 'r')
    all_data = f1.readlines()


    f2 = open("./Datasets/substrate_classes_all.csv", 'r')
    next(f2)
    all_tags = f2.readlines()

    X, y = [], []

    for i in range(0, len(all_data)):
        for tag in all_tags:
            if tag.split(',')[0] == all_data[i].strip('\n').strip('>'):
                all_data[i + 1].replace('U', 'X')
                all_data[i + 1].replace('Z', 'X')
                all_data[i + 1].replace('O', 'X')
                all_data[i + 1].replace('B', 'X')
                X.append(all_data[i + 1].strip('\n'))
                y.append(int(tag.split(',')[1])-1)
    return X,y