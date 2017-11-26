import numpy as np
import sys
import csv
import torch
from torch import autograd
import time
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model", help = "Select the Model you wish to train (cnn6, cnn29 or cnn54)")
parser.add_argument("dir", help = "Give the directory containing the dataset")
parser.add_argument("weights", help = "Give the path to the weights file")
args = parser.parse_args()

char_seq_length = 1024

if args.model == 'cnn6':
    import cnn6.model as model
    char_seq_length = 1014
elif args.model == 'cnn29':
    import cnn29.model as model
elif args.model == 'cnn54':
    import cnn54.model as model
else:
    print("Invalid model name")

alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/|_#$%ˆ&*˜‘+=<>()[]{} '

char_dict = {}
for i,c in enumerate(alphabet):
    char_dict[c] = i

#Function to load the dataset
def load_csv_file(filename, num_classes):
    all_data = []
    labels = []
    with open(filename) as f:
        reader = csv.DictReader(f,fieldnames=['class'],restkey='fields')
        for row in reader:
            # One-hot
            one_hot = np.zeros(num_classes)
            one_hot[int(row['class']) - 1] = 1
            labels.append(one_hot)
            # Text
            data = np.ones(char_seq_length) * 68
            text = row['fields'][1].lower()
            text = text[:min(char_seq_length, len(text))]
            for i in range(0, len(text)):
                if text[i] in char_dict:
                    data[i] = char_dict[text[i]]
                else:
                    data[i] = 67
                if i > char_seq_length - 1:
                    break
            all_data.append(data)
    f.close()
    return all_data, labels


# Function to call load_csv_file and return the output as arrays
def load_dataset():
    with open(args.dir + "classes.txt") as f:
        classes = []
        for line in f:
            classes.append(line.strip())
    f.close()
    num_classes = len(classes)
    train_data, train_label = [], []
    train_data, train_label = load_csv_file(args.dir + 'shuffled_train.csv', num_classes)
    test_data, test_label = load_csv_file(args.dir +'test.csv', num_classes)
    return train_data, train_label, test_data, test_label


# Function to create one-hot character vectors from the input string
def get_batched_one_hot(char_seqs_indices, labels, start_index, end_index):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"
    x_batch = char_seqs_indices[start_index:end_index]
    y_batch = labels[start_index:end_index]
    x_batch_one_hot = np.zeros(shape=[len(x_batch), len(alphabet), len(x_batch[0]), 1])
    start_index = int(start_index)
    end_index = int(end_index)
    for example_i, char_seq_indices in enumerate(x_batch):
        for char_pos_in_seq, char_seq_char_ind in enumerate(char_seq_indices):
            if char_seq_char_ind != -1:
                # print(i, example_i, char_seq_char_ind, char_pos_in_seq)
                example_i = int(example_i)
                char_seq_char_ind = int(char_seq_char_ind)
                char_pos_in_seq = int(char_pos_in_seq)
                x_batch_one_hot[example_i][char_seq_char_ind][char_pos_in_seq][0] = 1
    # print(x_batch_one_hot.shape)
    return [x_batch_one_hot, y_batch]


# Function that performs the training
def test_net(x, y, batch_size = 128, num_epochs = 100, shuffle=False):

    net = model.Model(14).cuda()
    net.load_state_dict(torch.load(args.weights)) 

    data_size = len(x)
    num_batches = int(data_size/batch_size)
    count = 0
    ncorrect = 0
    st = time.time()
    final_loss = 0

    for batch_num in range(num_batches):        
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        x_batch, y_batch = get_batched_one_hot(x, y, start_index, end_index)
        batch = torch.FloatTensor(x_batch).view(len(x_batch), 70, char_seq_length).cuda()
        batch = autograd.Variable(batch)

        out = net.forward(batch).cuda()

        _, pred = out.max(1)
        pred = pred.data
        pred = pred.cpu()
        pred = pred.numpy()
        target = np.argmax(y_batch, axis = 1)
        target = np.float32(target)
        for i in range(len(target)):
            if target[i] == pred[i]:
                ncorrect += 1
            count = count + 1

        target = autograd.Variable(torch.FloatTensor(target).cuda()).long()
        if batch_num%100 == 0:
            print("Batch Number : " + str(batch_num))
            print(str(ncorrect/count))

    print("Number of Correct Predictions : ", ncorrect)
    

if __name__ == '__main__':
    cudnn.benchmark = True
    train_data, train_label, test_data, test_label = load_dataset()
    test_net(test_data, test_label)
