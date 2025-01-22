import torch
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import time as t
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
from torchvision import utils

SAVE_PER_TIMES = 100
from GAN.Discriminator import Discriminator

class WGAN_GP(object):
    def __init__(self,channels):
        print("WGAN_GradientPenalty init model.")
        self.D = Discriminator(channels)
        self.C = channels
        # Check if cuda is available
        # self.check_cuda(args.cuda)

        # WGAN values from paper
        self.learning_rate = 1e-4
        self.b1 = 0.5
        self.b2 = 0.999

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))

        # Set the logger
        # self.logger = Logger('./logs')
        # self.logger.writer.flush()
        self.number_of_images = 10

        self.critic_iter = 5
        self.lambda_term = 10

    def train_generator(self,fake_images,weight_dtype):
        self.device = fake_images.device
        self.D = self.D.to(self.device, dtype=weight_dtype)
        fake_images = fake_images.to(dtype=weight_dtype)
        one = torch.tensor(1, dtype=weight_dtype)
        mone = one * -1
        mone = mone.to(self.device)
            
        # Generator update
        for p in self.D.parameters():
            p.requires_grad = False  # to avoid computation

        # train generator
        # compute loss with fake images
        # z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
        g_loss = self.D(fake_images)
        g_loss = g_loss.mean()
        g_loss.backward(mone,retain_graph=True)
        print(f'Generator iteration, g_loss: {-g_loss}') # -g_loss close to 1 is best
        return -g_loss
    def train_discriminator(self, real_images,fake_images,weight_dtype):
        self.device = real_images.device
        self.batch_size = real_images.shape[0]
        one = torch.tensor(1, dtype=weight_dtype)
        mone = one * -1
        one = one.to(self.device)
        mone = mone.to(self.device)
        self.D = self.D.to(self.device, dtype=weight_dtype)
        real_images = real_images.to(dtype=weight_dtype)
        fake_images = fake_images.to(dtype=weight_dtype)
        # Requires grad, Generator requires_grad = False
        for p in self.D.parameters():
            p.requires_grad = True

        # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
        self.D.zero_grad()

        # Train discriminator
        # WGAN - Training discriminator more iterations than generator
        # Train with real images
        d_loss_real = self.D(real_images)
        d_loss_real = d_loss_real.mean()
        d_loss_real.backward(mone)

        # Train with fake images
        d_loss_fake = self.D(fake_images)
        d_loss_fake = d_loss_fake.mean()
        d_loss_fake.backward(one)

        # Train with gradient penalty
        gradient_penalty = self.calculate_gradient_penalty(real_images.data, fake_images.data)
        gradient_penalty.backward()

        d_loss = d_loss_fake - d_loss_real + gradient_penalty
        Wasserstein_D = d_loss_real - d_loss_fake
        self.d_optimizer.step()
        # print(
        #     f'  Discriminator iteration, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')
        return d_loss, Wasserstein_D


    def evaluate(self, test_loader, D_model_path, G_model_path):
        self.load_model(D_model_path, G_model_path)
        z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = utils.make_grid(samples)
        print("Grid of 8x8 images saved to 'dgan_model_image.png'.")
        utils.save_image(grid, 'dgan_model_image.png')

    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(self.batch_size, 1, 1, 1).uniform_(0, 1)
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        eta = eta.to(self.device)


        interpolated = eta * real_images + ((1 - eta) * fake_images)

        interpolated = interpolated.to(self.device,dtype=real_images.dtype)


        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                  grad_outputs=torch.ones(
                                      prob_interpolated.size()).to(self.device),
                                  create_graph=True, retain_graph=True)[0]

        # flatten the gradients to it calculates norm batchwise
        gradients = gradients.view(gradients.size(0), -1)

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty

    def real_images(self, images, number_of_images):
        if (self.C == 3):
            return self.to_np(images.view(-1, self.C, 32, 32)[:self.number_of_images])
        else:
            return self.to_np(images.view(-1, 32, 32)[:self.number_of_images])

    def generate_img(self, z, number_of_images):
        samples = self.G(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            if self.C == 3:
                generated_images.append(sample.reshape(self.C, 32, 32))
            else:
                generated_images.append(sample.reshape(32, 32))
        return generated_images

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self):
        torch.save(self.G.state_dict(), './generator.pkl')
        torch.save(self.D.state_dict(), './discriminator.pkl')
        print('Models save to ./generator.pkl & ./discriminator.pkl ')

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))

    def get_infinite_batches(self, data_loader):
        while True:
            for i, (images, _) in enumerate(data_loader):
                yield images

