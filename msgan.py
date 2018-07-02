import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.utils as vutils
from tensorboardX import SummaryWriter


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
            n_input=100,
            n_out=(28 * 28),
            hidden_sizes=[256, 512, 1024],
            loss=nn.BCELoss()
    ):
        super(GeneratorNet, self).__init__()
        self.n_input = n_input
        self.n_out = n_out
        self.hidden_sizes = hidden_sizes
        self.loss = loss

        self.hidden0 = nn.Sequential(
            nn.Linear(n_input, hidden_sizes[0]),
            nn.LeakyReLU(0.2),
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.LeakyReLU(0.2),
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
        return error, prediction


class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(
        self,
        n_input=(28 * 28),
        n_out=1,
        hidden_sizes=[1024, 512, 256],
        loss=nn.BCELoss()
    ):
        super(DiscriminatorNet, self).__init__()
        self.n_input = n_input
        self.n_out = n_out
        self.hidden_sizes = hidden_sizes
        self.loss = loss

        self.hidden0 = nn.Sequential(
            nn.Linear(n_input, hidden_sizes[0]),
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
        real_data = real_data.view(real_data.size(0), self.n_input)
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
        return error_real, error_fake


class GAN():

    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, comment='GAN'):
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.writer = SummaryWriter(comment=comment)
        self.comment = comment

    def _images_to_vectors(self, images):
        return images.view(images.size(0), 784)

    def _vectors_to_images(self, vectors):
        return vectors.view(vectors.size(0), 1, 28, 28)

    def fit(self, data_generator, noise_generator, num_epochs=100):
        step = 0
        for epoch in range(num_epochs):
            for n_batch, (real_batch, _) in enumerate(data_generator):

                # 1. Train Discriminator
                real_data = Variable(real_batch)
                if torch.cuda.is_available():
                    real_data = real_data.cuda()
                # Generate fake data
                fake_data = self.generator(noise_generator(real_data.size(0))).detach()
                # Train D
                loss_real, loss_fake = self.discriminator.fit(self.d_optimizer, real_data, fake_data)

                # 2. Train Generator
                # Generate fake data
                fake_data = self.generator(noise_generator(real_batch.size(0)))
                # Train G
                g_loss, _ = self.generator.fit(self.g_optimizer, fake_data, self.discriminator)
                step += 1
                self.log_scalars(loss_real, loss_fake, g_loss, step)

            test_noise = noise_generator(16)
            images = self.generator(test_noise)
            self.log_images(images, step)

            print(f'Epoch: [{epoch}/{num_epochs}]')
            print(f'Generator Loss: {g_loss:.4f}')
            print(f'D(x): {loss_real.mean():.4f}, D(G(z)): { loss_fake.mean():.4f}')


    def log_scalars(self, loss_real, loss_fake, g_loss, step):
        self.writer.add_scalar(f'{self.comment}/D_error', loss_real + loss_fake, step)
        self.writer.add_scalar(f'{self.comment}/G_error', g_loss, step)
        self.writer.add_scalar(f'{self.comment}/D(x)', loss_real.mean(), step)
        self.writer.add_scalar(f'{self.comment}/D(g(y))', loss_fake.mean(), step)

    def log_images(self, images, step):
        images = self._vectors_to_images(images).data.cpu()
        # images = torch.from_numpy(test_images)
        img_name = '{}/images{}'.format(self.comment, '')

        # Make horizontal grid from image tensor
        horizontal_grid = vutils.make_grid(images, scale_each=True)
        # Add horizontal images to tensorboard
        self.writer.add_image(img_name, horizontal_grid, step)

#
# class GAN_ORIGINAL():
#
#     def __init__(self, discriminator, generator, d_optimizer, g_optimizer, name='GAN ORIGINAL'):
#         self.discriminator = discriminator
#         self.generator = generator
#         self.d_optimizer = d_optimizer
#         self.g_optimizer = g_optimizer
#         if torch.cuda.is_available():
#             self.discriminator = discriminator.cuda()
#             self.generator = generator.cuda()
#
#         self.logger = Logger(model_name=name, data_name='MNIST')
#
#     def _real_data_target(self, size):
#         '''
#         Tensor containing ones, with shape = size
#         '''
#         data = Variable(torch.ones(size, 1))
#         if torch.cuda.is_available():
#             return data.cuda()
#         return data
#
#     def _fake_data_target(self, size):
#         '''
#         Tensor containing zeros, with shape = size
#         '''
#         data = Variable(torch.zeros(size, 1))
#         if torch.cuda.is_available():
#             return data.cuda()
#         return data
#
#
#     def _images_to_vectors(self, images):
#         return images.view(images.size(0), 784)
#
#     def _vectors_to_images(self, vectors):
#         return vectors.view(vectors.size(0), 1, 28, 28)
#
#
#     def _train_discriminator(self, optimizer, real_data, fake_data, loss):
#         # Reset gradients
#         optimizer.zero_grad()
#
#         # 1.1 Train on Real Data
#         prediction_real = self.discriminator(real_data)
#         # Calculate error and backpropagate
#         error_real = loss(prediction_real, self._real_data_target(real_data.size(0)))
#         error_real.backward()
#
#         # 1.2 Train on Fake Data
#         prediction_fake = self.discriminator(fake_data)
#         # Calculate error and backpropagate
#         error_fake = loss(prediction_fake, self._fake_data_target(real_data.size(0)))
#         error_fake.backward()
#
#         # 1.3 Update weights with gradients
#         optimizer.step()
#
#         # Return error
#         return error_real + error_fake, prediction_real, prediction_fake
#
#     def _train_generator(self, optimizer, fake_data, loss):
#         # 2. Train Generator
#         # Reset gradients
#         optimizer.zero_grad()
#         # Sample noise and generate fake data
#         prediction = self.discriminator(fake_data)
#         # Calculate error and backpropagate
#         error = loss(prediction, self._real_data_target(prediction.size(0)))
#         error.backward()
#         # Update weights with gradients
#         optimizer.step()
#         # Return error
#         return error
#
#     def fit(self, data_loader, input_noise, loss=nn.BCELoss(), num_epochs=100):
#         num_batches = len(data_loader)
#         for epoch in range(num_epochs):
#             for n_batch, (real_batch, _) in enumerate(data_loader):
#
#                 # 1. Train Discriminator
#                 real_data = Variable(self._images_to_vectors(real_batch))
#
#                 if torch.cuda.is_available():
#                     real_data = real_data.cuda()
#
#                 # Generate fake data
#                 fake_data = self.generator(input_noise(real_data.size(0))).detach()
#                 # Train D
#                 d_error, d_pred_real, d_pred_fake = self._train_discriminator(self.d_optimizer, real_data, fake_data, loss)
#
#                 # 2. Train Generator
#                 # Generate fake data
#                 fake_data = self.generator(input_noise(real_batch.size(0)))
#                 # Train G
#                 g_error = self._train_generator(self.g_optimizer, fake_data, loss)
#                 # Log error
#                 self.logger.log(d_error, g_error, d_pred_real, d_pred_fake, epoch, n_batch, num_batches)
#
#             # Display Progress
# #             display.clear_output(True)
#             num_test_samples = 16
#             test_noise = input_noise(num_test_samples)
#             test_images = self._vectors_to_images(self.generator(test_noise)).data.cpu()
#             self.logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches)
#             self.logger.display_status(epoch, num_epochs, d_error, g_error, d_pred_real, d_pred_fake)
#             self.logger.save_models(self.generator, self.discriminator, epoch)
