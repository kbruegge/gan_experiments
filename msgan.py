import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def real_data_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    if torch.cuda.is_available():
        return data.cuda()
    return data


def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available():
        return data.cuda()
    return data


class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(
            self,
            n_features=100,
            n_out=(28 * 28),
            hidden_sizes=[256, 512, 1024],
            loss=nn.BCELoss()
    ):
        super(GeneratorNet, self).__init__()
        self.n_features = n_features
        self.n_out = n_out
        self.hidden_sizes = hidden_sizes
        self.loss = loss

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, hidden_sizes[0]),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.Sigmoid()
        )

        self.out = nn.Sequential(
            nn.Linear(hidden_sizes[2], n_out),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

    def fit(self, optimizer, fake_data, discriminator):
        # 2. Train Generator
        # Reset gradients
        optimizer.zero_grad()
        # Sample noise and generate fake data
        prediction = discriminator(fake_data)
        # Calculate error and backpropagate
        error = self.loss(prediction, real_data_target(prediction.size(0)))
        error.backward()
        # Update weights with gradients
        optimizer.step()
        # Return error
        return error


class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(
        self,
        n_features=(28 * 28),
        n_out=1,
        hidden_sizes=[1024, 512, 256],
        loss=nn.BCELoss()
    ):
        super(DiscriminatorNet, self).__init__()
        self.n_features = n_features
        self.n_out = n_out
        self.hidden_sizes = hidden_sizes
        self.loss = loss

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, hidden_sizes[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(hidden_sizes[2], n_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


    def fit(self, optimizer, real_data, fake_data):
        # Reset gradients
        optimizer.zero_grad()

        # 1.1 Train on Real Data
        real_data = real_data.view(real_data.size(0), self.n_features)
        prediction_real = self(real_data)
        # Calculate error and backpropagate
        error_real = self.loss(prediction_real, real_data_target(real_data.size(0)))
        error_real.backward()

        # 1.2 Train on Fake Data
        prediction_fake = self(fake_data)
        # Calculate error and backpropagate
        error_fake = self.loss(prediction_fake, fake_data_target(real_data.size(0)))
        error_fake.backward()

        # 1.3 Update weights with gradients
        optimizer.step()

        # Return error
        return error_real + error_fake, prediction_real, prediction_fake


class Generator50(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(Generator50, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.sigmoid(self.map2(x))
        return self.map3(x)


class Discriminator50(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, output_size=1):
        super(Discriminator50, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.elu(self.map2(x))
        return F.sigmoid(self.map3(x))


class GAN50():

    def extract(self, v):
        return v.data.storage().tolist()

    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, loss_function=nn.BCELoss()):
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_function = loss_function

    def fit(self, real_data_sampler, fake_data_sampler, d_steps=1, g_steps=1, minibatch_size=100, print_interval=2000):
        D = self.discriminator
        G = self.generator
        for d_index in range(d_steps):
            # 1. Train D on real+fake
            D.zero_grad()

            #  1A: Train D on real

            d_real_data = Variable(real_data_sampler(D.input_size))
            d_real_decision = D(d_real_data)
            d_real_error = self.loss_function(d_real_decision, Variable(torch.ones(1, 1)))  # ones = true
            d_real_error.backward() # compute/store gradients, but don't change params

            #  1B: Train D on fake
            d_gen_input = Variable(fake_data_sampler(minibatch_size, G.input_size))
            d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
            d_fake_decision = D(d_fake_data.t())
            d_fake_error = self.loss_function(d_fake_decision, Variable(torch.zeros(1, 1)))  # zeros = fake
            d_fake_error.backward()
            self.d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

        for g_index in range(g_steps):
            # 2. Train G on D's response (but DO NOT train D on these labels)
            G.zero_grad()

            gen_input = Variable(fake_data_sampler(minibatch_size, G.input_size))
            g_fake_data = G(gen_input)
            dg_fake_decision = D(g_fake_data.t())
            g_error = self.loss_function(dg_fake_decision, Variable(torch.ones(1, 1)))  # we want to fool, so pretend it's all genuine

            g_error.backward()
            self.g_optimizer.step()  # Only optimizes G's parameters

        return float(d_real_error), float(d_fake_error), float(g_error), np.array(d_real_data), np.array(d_fake_data.data)


class MSGAN():

    def __init__(self, generator, discriminator, g_optimizer, d_optimizer):
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer


    def fit(self, data_generator, noise_generator, num_epochs=100):
        for n_batch, real_batch in enumerate(data_generator):

            # 1. Train Discriminator
            real_data = Variable(real_batch)
            if torch.cuda.is_available():
                real_data = real_data.cuda()
            # Generate fake data
            fake_data = self.generator(noise_generator(real_data.size(0))).detach()
            # Train D
            self.discriminator.fit(self.d_optimizer, real_data, fake_data)

            # 2. Train Generator
            # Generate fake data
            fake_data = self.generator(noise_generator(real_batch.size(0)))
            # Train G
            self.generator.fit(self.g_optimizer, fake_data, self.discriminator)
