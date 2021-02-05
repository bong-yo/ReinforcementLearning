from torch import FloatTensor, LongTensor


def create_batches(data, size, randomize=False):
    '''Return inputs and targets organized in batches.'''
    n_batches = len(data) // size + int(len(data) % size != 0)
    for i in range(n_batches):
        yield zip(*data[i * size: (i + 1) * size])


def unzip_to_tensors(batch):
    tensors = []
    for feature in zip(*batch):
        if type(feature[0]) == int:
            tensors.append(LongTensor(feature))
        else:
            tensors.append(FloatTensor(feature))

    return tensors
