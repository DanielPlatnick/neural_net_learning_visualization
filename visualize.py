# import pygame module in this program
import pygame
import sys
import time
import random
import math
import numpy as np
from pygame.locals import *
from pygame.color import Color

# activate the pygame library .
# initiate pygame and give permission
# to use pygame's functionality.
pygame.init()
pygame.font.init()

class Setting:
    def __init__(self):
        self.screen_width = 1000
        self.screen_height = 1000

class Neuron:

    def __init__(self, x, y, radius, color, value, font_size=5):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.value = value
        self.font_size = font_size
    
    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius, 1)
        # render a text written in this font inside the circle
        # make text font and size dynamic
        # font size is dependent on the radius of the circle
        font_size = self.radius * 1.5
        font = pygame.font.SysFont('Comic Sans MS', int(font_size))
        text = font.render(str(np.round(self.value, 2)), True, self.color)
        textRect = text.get_rect()
        textRect.center = (self.x, self.y)
        screen.blit(text, textRect)

    
    def update(self):
        # update the value of the neuron
        pass

class Layer:

    def __init__(self, x, y, num_neurons, radius, color, isinput=False, value=None):
        self.x = x
        self.y = y
        self.num_neurons = num_neurons
        self.radius = radius
        self.color = color
        self.neurons = []
        y_margin = 10
        # add y_margin to the radius to make sure the neurons don't overlap
        y_step = (self.radius + y_margin) * 2
        self.values = value
        for i in range(self.num_neurons):
            # if it is the input layer, the value is the input
            if isinput:
                value = self.values[i]
            else:
                value = 0
            self.neurons.append(Neuron(self.x, self.y + y_step * i, self.radius, self.color, value))
            
    def draw(self, screen):
        for neuron in self.neurons:
            neuron.draw(screen)
    
    def update(self):
        # get the neuron of previous layer and next layer
        # update the value of the neuron
        for neuron in self.neurons:
            neuron.update()

class Network:
    """
    Customize number of layers and neurons
    """
    
    def __init__(self, x, y, num_layers, radius, color,*custom_neuron_each_layer):
        self.x = x
        self.y = y
        self.num_layers = num_layers
        self.radius = radius
        self.color = color
        self.layers = []
        self.weights = []
        x_margin = 10
        # add x_margin to the radius to make sure the neurons don't overlap
        x_step = (self.radius + x_margin) * 10
        self.num_neurons = []

        index = 0
        for neuron in custom_neuron_each_layer:
            self.weights.append([])
            self.num_neurons.append(neuron)
            # adjust layer y postion depedning on the number of neurons
            y = self.y - (neuron * (self.radius + 10) * 2) / 2
            # if it is the input layer, the value is the input
            # add weights for each layer
            if index == 0:
                # add weights for the input layer
                # the input layer has no weights
                self.weights[index].append([])
                self.layers.append(Layer(self.x + x_step * index, y, neuron, self.radius, self.color, True, np.random.rand(neuron)))
            else:
                # add weights for the hidden layer
                # the input layer has no weights
                self.weights[index].append(np.random.rand(self.num_neurons[index - 1] + 1, neuron))
                self.layers.append(Layer(self.x + x_step * len(self.layers), y, neuron, self.radius, self.color))
            index += 1

    def draw(self, screen):
        for layer in self.layers:
            layer.draw(screen)
        # draw lines
        for i in range(self.num_layers - 1):
            for neuron1 in self.layers[i].neurons:
                for neuron2 in self.layers[i + 1].neurons:
                    # have different color for each batch of lines
                    # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    pygame.draw.line(screen, (0, 0, 0), (neuron1.x, neuron1.y), (neuron2.x, neuron2.y), 1)
                    
    def update(self):
        #feed forward the network and update the neurons
        # multiply the weights with the input
        # add the bias
        # apply the activation function
        # update the value of the neuron
        # simulate the network
        # show the updated value of the neuron every 0.5 seconds
        for i in range(self.num_layers - 1):
            for j in range(len(self.layers[i + 1].neurons)):
                # get the value of the neuron in the previous layer
                value = 0
                # get previous layer neurons
                for k in range(len(self.layers[i].neurons)):
                    value += self.layers[i].neurons[k].value * self.weights[i + 1][0][k][j]

                # add the bias
                # value += self.weights[i][len(self.layers[i].neurons)][j]
                # apply the activation function
                value = self.activation_function(value)
                self.layers[i + 1].neurons[j].value = value



    def activation_function(self, value):
        # apply the activation function
        return 1 / (1 + np.exp(-value))

class Visulize:
    def __init__(self):
        self.scene = []
        self.setting = Setting()
        self.screen = pygame.display.set_mode((self.setting.screen_width, self.setting.screen_height))

        self.awake()
    
    def awake(self):
        # self.scene.append(Network(100, 530, 4, 15, (0, 0, 255),20, 10, 5, 3))
        self.scene.append(Network(100, 530, 3, 15, (0, 0, 255),4, 3, 2))


    def update(self):
        for i in self.scene:
            i.update()
            i.draw(self.screen)
        pygame.display.update()
        self.screen.fill(Color('white'))

    def run(self):
        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
            self.update()
        self.quit()

    def quit(self):
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    visulize = Visulize()
    visulize.run()