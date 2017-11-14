import numpy as np
import json
import sys
import model
import torch
from torch import autograd
import time
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import pickle 

def load_yelp(alphabet):
    examples = []
    labels = []
    filename = "../dataset/test.json"
    with open(filename) as f:
        i = 0
        for line in f:
            review = json.loads(line)
            stars = review["stars"]
            text = review["text"]
            if stars != 3:
                text_end_extracted = extract_end(list(text.lower()))
                padded = pad_sentence(text_end_extracted)
                text_int8_repr = string_to_int8_conversion(padded, alphabet)
                if stars == 1 or stars == 2:
                    labels.append([1, 0])
                    examples.append(text_int8_repr)
                elif stars == 4 or stars == 5:
                    labels.append([0, 1])
                    examples.append(text_int8_repr)
                i += 1
                if i%10000 == 0:
                    print(i)
                    return examples, labels
    return examples, labels


def extract_end(char_seq):
    if len(char_seq) > 1014:
        char_seq = char_seq[-1014:]
    return char_seq


def pad_sentence(char_seq, padding_char=" "):
    char_seq_length = 1014
    num_padding = char_seq_length - len(char_seq)
    new_char_seq = char_seq + [padding_char] * num_padding
    return new_char_seq


def string_to_int8_conversion(char_seq, alphabet):
    x = np.array([alphabet.find(char) for char in char_seq], dtype=np.int8)
    return x


def get_batched_one_hot(char_seqs_indices, labels, start_index, end_index):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"
    x_batch = char_seqs_indices[start_index:end_index]
    y_batch = labels[start_index:end_index]
    x_batch_one_hot = np.zeros(shape=[len(x_batch), len(alphabet), len(x_batch[0]), 1])
    for example_i, char_seq_indices in enumerate(x_batch):
        for char_pos_in_seq, char_seq_char_ind in enumerate(char_seq_indices):
            if char_seq_char_ind != -1:
                x_batch_one_hot[example_i][char_seq_char_ind][char_pos_in_seq][0] = 1
    return [x_batch_one_hot, y_batch]


def load_data():

    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"
    examples, labels = load_yelp(alphabet)
    x = np.array(examples, dtype=np.int8)
    y = np.array(labels, dtype=np.int8)
    print("x_char_seq_ind=" + str(x.shape))
    print("y shape=" + str(y.shape))
    return [x, y]


def batch_iter(x, y, batch_size = 10, num_epochs = 100, shuffle=False):

    net = model.Model(2).cuda()
    net.load_state_dict(torch.load('train_backup_small_again14.pt'))
    data_size = len(x)
    num_batches = int(data_size/batch_size)
    ncorrect = 0
    for batch_num in range(num_batches):
        # if batch_num%100 == 0:
        print("Batch Number : " + str(batch_num))
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        x_batch, y_batch = get_batched_one_hot(x, y, start_index, end_index)
        batch = torch.FloatTensor(x_batch).view(len(x_batch), 70, 1014).cuda()
        batch = autograd.Variable(batch)

        out = net.forward(batch).cuda()
        _, pred = out.max(1)
        pred = pred.data
        pred = pred.cpu()
        pred = pred.numpy()
        print(pred)
        target = y_batch[:,1]
        target = np.float32(target)
        # target = torch.FloatTensor(target).cuda()
    
        for i in range(len(target)):
            x = pred[i]
            if target[i] == x:
                ncorrect += 1
    print(str(ncorrect/data_size))

def batch_iter(x, y, batch_size = 10, num_epochs = 100, shuffle=False):

    net = model.Model(2).cuda()
    net.load_state_dict(torch.load('train_backup_small3.pt'))
    # loss = torch.nn.CrossEntropyLoss().cuda()
    # optimizer = optim.Adam(net.parameters(), lr=0.001)
    # print("TEST")
    data_size = len(x)
    num_batches = int(data_size/batch_size)
    count = 0
    ncorrect, FP, FN = 0, 0, 0
    # f = str('train_backup.pt')
    # for epoch in range(num_epochs):
    st = time.time()
    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]
    else:
        x_shuffled = x
        y_shuffled = y

    final_loss = 0
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        x_batch, y_batch = get_batched_one_hot(x_shuffled, y_shuffled, start_index, end_index)
        batch = torch.FloatTensor(x_batch).view(len(x_batch), 70, 1014).cuda()
        batch = autograd.Variable(batch)

        # optimizer.zero_grad()

        out = net.forward(batch).cuda()
        _, pred = out.max(1)
        pred = pred.data
        pred = pred.cpu()
        pred = pred.numpy()
        # print(pred)
        target = y_batch[:,1]
        target = np.float32(target)
        # print(target)
        # target = torch.FloatTensor(target).cuda()
        for i in range(len(target)):
            if target[i] == pred[i]:
                ncorrect += 1
            elif target[i] == 0:
                FP += 1
            else:
                FN += 1
            count = count + 1
        target = autograd.Variable(torch.FloatTensor(target).cuda()).long()
        if batch_num%10 == 0:
            print("Batch Number : " + str(batch_num))
            print(str(ncorrect/count))
    print("Number of Correct Predictions : ", ncorrect)
    print("Number of False Positives : ", FP)
    print("Number of False Negatives : ", FN)

[x,y] = load_data()
cudnn.benchmark = True
batch_iter(x, y)