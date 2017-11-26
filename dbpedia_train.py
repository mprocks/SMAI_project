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


# Function to shuffle the dataset
def shuffle_dataset():
    fid = open(args.dir + 'train.csv', "r")
    li = fid.readlines()
    fid.close()
    random.shuffle(li)
    fid = open(args.dir + 'shuffled_train.csv', "w")
    fid.writelines(li)
    fid.close()


# Function to call load_csv_file and return the output as arrays
def load_dataset():
    with open(args.dir + "classes.txt") as f:
        classes = []
        for line in f:
            classes.append(line.strip())
    f.close()
    shuffle_dataset()
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
                example_i = int(example_i)
                char_seq_char_ind = int(char_seq_char_ind)
                char_pos_in_seq = int(char_pos_in_seq)
                x_batch_one_hot[example_i][char_seq_char_ind][char_pos_in_seq][0] = 1
    return [x_batch_one_hot, y_batch]


# Function that performs the training
def train_net(x, y, batch_size = 128, num_epochs = 6):
    # Limiting size of dataset
    x = x[:200000]
    y = y[:200000]

    # Creating output log file
    outfile = open("outfile.txt", "w")

    # Loading the model
    net = model.Model(14).cuda()

    # Initializing loss function and optimizer
    loss = torch.nn.CrossEntropyLoss().cuda()
    learning_rate = 10**(-3)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    
    data_size = len(x)
    print("SIZE = ", data_size)
    num_batches = int(data_size/batch_size)
    count = 0

    for epoch in range(num_epochs):

        st = time.time()
        print("\n\n\n\n\nIn epoch >> " + str(epoch + 1))
        print("num batches per epoch is: " + str(num_batches))

        final_loss = 0        
        for batch_num in range(num_batches):

            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            x_batch, y_batch = get_batched_one_hot(x, y, start_index, end_index)
            batch = torch.FloatTensor(x_batch).view(len(x_batch), 70, char_seq_length).cuda()
            batch = autograd.Variable(batch)

            optimizer.zero_grad()

            out = net.forward(batch).cuda()

            _, pred = out.max(1)
            target = np.argmax(y_batch, axis = 1)
            target = np.float32(target)    
            target = autograd.Variable(torch.FloatTensor(target).cuda()).long()
            output = loss(out, target).cuda()
            output.backward()
            optimizer.step()

            count = count + 1
            final_loss += output.data[0]

            if batch_num%100 == 0:
                print("Batch Number : " + str(batch_num))
                print("Loss : ", final_loss/(batch_num+1))
                outfile.write("Batch Number : " + str(batch_num))
                outfile.write("Loss : " + str(final_loss/(batch_num+1)))

            if((batch_num+1)%(int(num_batches/2)) == 0):
                optimizer = optim.Adam(net.parameters(), lr=learning_rate)
                learning_rate /= 2
                outfile.write("Learning rate : " + str(learning_rate))

        print("Time per epoch = " + str(time.time() - st))
        outfile.write("Loss after epoch " + str(epoch) + " = " + str(final_loss/num_batches))
        print("Loss after epoch " + str(epoch) + " = " + str(final_loss/num_batches))

        # Saving the model
        torch.save(net.state_dict(), str('train_backup_small_again' + str(epoch) + '.pt'))
        
    outfile.close()

if __name__ == '__main__':
    cudnn.benchmark = True
    train_data, train_label, test_data, test_label = load_dataset()
    train_net(train_data, train_label)
