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

class Setting:
    def __init__(self):
        self.screen_width = 1000
        self.screen_height = 1000

class Neuron:

    def __init__(self, x, y, radius, color, value):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.value = value
    
    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius, 1)
    
    def update(self):
        self.value = value
        # self.color = (0, 0, 255 * value)

class Layer:

    def __init__(self, x, y, num_neurons, radius, color):
        self.x = x
        self.y = y
        self.num_neurons = num_neurons
        self.radius = radius
        self.color = color
        self.neurons = []
        y_margin = 10
        # add y_margin to the radius to make sure the neurons don't overlap
        y_step = (self.radius + y_margin) * 2
        for i in range(self.num_neurons):
            self.neurons.append(Neuron(self.x, self.y + y_step * i, self.radius, self.color, 0))
            
    def draw(self, screen):
        for neuron in self.neurons:
            neuron.draw(screen)
    
    def update(self):
        # for i in range(self.num_neurons):
            # self.neurons[i].update(values[i])
        pass

class Network:
    """
    Customize number of layers and neurons
    """
    
    def __init__(self, x, y, num_layers, radius, color,*cusom_neuroon_each_layer):
        self.x = x
        self.y = y
        self.num_layers = num_layers
        self.radius = radius
        self.color = color
        self.layers = []
        x_margin = 10
        # add x_margin to the radius to make sure the neurons don't overlap
        x_step = (self.radius + x_margin) * 10
        self.num_neurons = []

        for neuron in cusom_neuroon_each_layer:
            self.num_neurons.append(neuron)
            self.layers.append(Layer(self.x + x_step * len(self.layers), self.y, neuron, self.radius, self.color))
        
    def draw(self, screen):
        for layer in self.layers:
            layer.draw(screen)
        # draw lines
        for i in range(self.num_layers - 1):
            for neuron1 in self.layers[i].neurons:
                for neuron2 in self.layers[i + 1].neurons:
                    pygame.draw.line(screen, (0, 0, 0), (neuron1.x, neuron1.y), (neuron2.x, neuron2.y), 1)
        
    def update(self):
        # for i in range(self.num_layers):
            # self.layers[i].update(values[i])
        pass

class Visulize:
    def __init__(self):
        self.scene = []
        self.setting = Setting()
        self.screen = pygame.display.set_mode((self.setting.screen_width, self.setting.screen_height))

        self.awake()
    
    def awake(self):
        self.scene.append(Network(100, 100, 4, 10, (0, 0, 255),10, 9, 8, 7))


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