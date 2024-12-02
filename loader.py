
import random
import pickle
from tqdm import tqdm

class MyLoader():
    def __init__(self, indices, shuffle=True, cap_size=None, path='Datasets/traincv/', cache=True):
        if cap_size is not None:
            indices = indices[:min(len(indices), cap_size)]

        self.n = len(indices)
        self.shuffle = shuffle
        self.data = []
        self.cache = cache
        self.path = path
        print("Loading dataset")
        self.indices = indices
        if self.cache:
            for i in tqdm(range(self.n)):
                file = open(path + str(self.indices[i]) + '.pkl', 'rb')
                self.data.append(pickle.load(file))
                file.close()
        self.order = [i for i in range(self.n)]
        self.current = 0
        random.shuffle(self.order)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.n:
            # file = open('traincv/' + str(self.indices[self.order[self.current]]) + '.pkl', 'rb')
            # data = pickle.load(file)
            # file.close()
            self.current += 1
            if self.cache:
                return self.data[self.order[self.current-1]]
            else:
                file = open(self.path + str(self.order[self.current-1]) + '.pkl', 'rb')
                return pickle.load(file)
        random.shuffle(self.order)
        self.current = 0
        raise StopIteration
    
    def __len__(self):
        return self.n