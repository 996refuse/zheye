from PIL import Image, ImageFont, ImageDraw
import numpy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def random_character():
    '''
    produce a random chinese character in unicode.
    '''
     
    head = random.randint(0xb0, 0xf7)
    if head == 0xd7:
        body = random.randint(0xa1, 0xf9)
    else:
        body = random.randint(0xa1, 0xfe)
    val = f"{head:x}{body:x}"
    return bytes.fromhex(val).decode("gb2312")


def character_to_image(c):
    '''
    convert a character to 40 * 40 image.
    '''
    
    angle = random.choice(range(-40, 40))
    scale = random.uniform(0.8, 1.2)
    width_err = random.randint(-10, 10)
    height_err = random.randint(-10, 10)
    ttf_path = "Kaiti-SC-Bold.ttf"

    im = Image.new("RGBA", (72, 102), (0, 0, 0, 0))
    font = ImageFont.truetype(ttf_path, 72)

    dr = ImageDraw.Draw(im)
    dr.fontmode = "1"
    dr.text((0, 0), c, font=font, fill="#000000")
    
    im = im.rotate(angle, expand=1)
    width, height = im.size
    im = im.resize((int(width *scale), int(height*scale)), Image.ANTIALIAS)
    width, height = im.size

    bg = Image.new("RGBA", (160, 160), (255,255,255,255))
    bg.paste(im, (int(80 - width/2 + width_err), int(80 - height/2 + height_err)), im)
    return bg.crop((60, 60, 100, 100))


def image_to_training_data(image):
    '''
    produce one sample given a image.
    1 means upright (positive sample),
    0 means upside down (negative sample).
    '''
    
    Y = random.choice([1, 0])
    X = numpy.asarray(image.convert("L")).astype("float32")
    
    X[X <= 150] = -1
    # black
    X[X >  150] = 1
    # white
    
    if Y == 0:
        X = numpy.rot90(X, 2)
    
    return X, Y


def data_to_image(d):
    '''
    convert 2darray to image object.
    '''
    
    return Image.fromarray(numpy.uint8(d))


def generate_a_batch(s):
    '''
    generate a mini batch with size s.
    '''
    
    inputs = []
    labels = []
    for i in range(s):
        c = random_character()
        image = character_to_image(c)
        X, Y = image_to_training_data(image)
        inputs.append(X)
        labels.append(Y)
    inputs = numpy.array(inputs)
    labels = numpy.array(labels)
    return inputs, labels


def generate_labeled_data(batch=10000, batch_size=100):
    '''
    generate labeled data.
    CPU consume high, very slow.
    '''
    
    data = []
    for i in range(batch):
        data.append(generate_a_batch(batch_size))
        print("%.3f%%" % ((i+1)*100.0/batch), end="\r")
    return data


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 25, 40)
        self.fc2 = nn.Linear(40, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def main():
    training_batch = 10
    testing_batch = 1
    batch_size = 10

    training_data = generate_labeled_data(training_batch, batch_size)
    testing_data = generate_labeled_data(testing_batch, batch_size)

    device = torch.device("cuda")
    net = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    print(net)

    print("neural network: training")

    while True:
        for i, data in enumerate(training_data):
            inputs, labels = data

            inputs = torch.from_numpy(inputs.reshape(batch_size,1,40,40))
            labels = torch.from_numpy(labels)

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            print('%3d%% loss: %.3f' %
                ((i + 1)/training_batch*100, loss.item()), end='\r')
            
        if loss.item() < 0.002:
            break

    print("neural network: testing")
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testing_data:
            inputs, labels = data
            
            inputs = torch.from_numpy(inputs.reshape(batch_size,1,40,40).copy())
            labels = torch.from_numpy(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the %d test inputs: %f %%' % (total, (
        100.0 * correct / total)))

    print("neural network: saving")
    torch.save(net, "./zheye.pt")


if __name__ == "__main__":
    main()
