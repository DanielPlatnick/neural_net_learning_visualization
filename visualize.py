# import pygame module in this program
from hashlib import new
import pygame
import sys
import time
import random
import math
from utils import *
from pygame.locals import *
from pygame.color import Color

from main import Model, train, torch, optim, TensorDataset, features, labels, DataLoader

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
    
    def __init__(self, x, y, num_layers, radius, color, chain:Chain, *custom_neuron_each_layer):
        self.x = x
        self.y = y
        self.num_layers = num_layers
        self.radius = radius
        self.color = color
        self.layers = []
        self.weights = []
        self.chain = chain
        x_margin = 10
        # add x_margin to the radius to make sure the neurons don't overlap
        x_step = (self.radius + x_margin) * 10
        self.num_neurons = []

        index = 0
        for neuron in custom_neuron_each_layer:
            self.num_neurons.append(neuron)
            # adjust layer y postion depedning on the number of neurons
            y = self.y - (neuron * (self.radius + 10) * 2) / 2
            # if it is the input layer, the value is the input
            # add weights for each layer
            if index == 0:
                # add weights for the input layer
                # the input layer has no weights
                # flatten features
                features_flatten = features.view(features.shape[0], -1)
                self.layers.append(Layer(self.x, y, neuron, self.radius, self.color, isinput=True, value=features_flatten.tolist()[0]))
            else:
                # add weights for the hidden layer
                # the input layer has no weights
                self.layers.append(Layer(self.x + x_step * len(self.layers), y, neuron, self.radius, self.color))
                # for i in range(neuron):
                    # the number of weights is equal to the number of neurons in the previous layer
                    # self.weights.append(Weight(self.layers[index - 1].neurons[i], self.layers[index].neurons[i], np.random.rand(1)[0]))

            index += 1
        # for i in range(len(self.layers)):
            # if i == 0:
                # self.weights.append(Weight(None, None, None))
            # else:
                # for j in range(len(self.layers[i].neurons)):
                    # self.weights.append(Weight(self.layers[i - 1].neurons[j], self.layers[i].neurons[j], np.random.rand(1)[0]))
        for i in range(self.num_layers - 1):
            for neuron1 in self.layers[i].neurons:
                for neuron2 in self.layers[i + 1].neurons:
                    self.weights.append(Weight(neuron1, neuron2, np.random.rand(1)[0]))

    def draw(self, screen):
        for layer in self.layers:
            layer.draw(screen)
        # draw lines
        for i in range(self.num_layers - 1):
            for neuron1 in self.layers[i].neurons:
                for neuron2 in self.layers[i + 1].neurons:
                    # have different color for each batch of lines
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    # color = (0,0,0)
                    pygame.draw.line(screen, color, (neuron1.x, neuron1.y), (neuron2.x, neuron2.y), 1)

        # render a text written in this font middle of the line
        # get neuron1 and neuron2 position and calculate the middle point
        # make text font and size dynamic
        # font size is dependent on the radius of the circle
        weight_font_size = self.radius*0.5
        weight_font = pygame.font.SysFont('Comic Sans MS', int(weight_font_size))
        for weight in self.weights:
            if weight.value is not None:
                text = weight_font.render(str(np.round(weight.value, 2)), True, (0, 0, 0))
                textRect = text.get_rect()
                textRect.center = (((7*weight.neuron1.x) + weight.neuron2.x) / 8, ((7*weight.neuron1.y) + weight.neuron2.y) / 8)
                screen.blit(text, textRect)
                        
                    
    def update(self):
        self.train()

    def train(self):
        # get the input neurons
        input_neurons = self.layers[0].neurons
        # feed forward
        hidden_layer_1 = self.layers[1].neurons
        hidden_layer_2 = self.layers[2].neurons
        output_layer = self.layers[3].neurons
        
        weights1 = []
        bias1 = []

        # get the weights
        for i in range(len(input_neurons) * len(hidden_layer_1)):
            # get the weight
            weights1.append(self.weights[i].value)
            # get the bias
            bias1.append(self.weights[i].value)
        # reshape the weights
        weights1 = np.array(weights1).reshape(len(input_neurons), len(hidden_layer_1))
        # reshape the bias
        bias1 = np.array(bias1).reshape(len(input_neurons), len(hidden_layer_1))

        # get input_neurons as numpy array
        neurons1 = np.array([neuron.value for neuron in input_neurons])

        output1  = self.chain[0](np.dot(neurons1, weights1) + bias1)
        
        # flatten the output
        output1 = output1.flatten()
        # update the value of the neurons
        for i in range(len(hidden_layer_1)):
            hidden_layer_1[i].value = output1[i]

        weights2 = []
        bias2 = []
        # get the weights
        for i in range(len(hidden_layer_1) * len(hidden_layer_2)):
            # get the weight
            weights2.append(self.weights[i + len(input_neurons) * len(hidden_layer_1)].value)
            # get the bias
            bias2.append(self.weights[i + len(input_neurons) * len(hidden_layer_1)].value)

        # reshape the weights
        weights2 = np.array(weights2).reshape(len(hidden_layer_1), len(hidden_layer_2))
        # reshape the bias
        bias2 = np.array(bias2).reshape(len(hidden_layer_1), len(hidden_layer_2))

        # get input_neurons as numpy array
        neurons2 = np.array([neuron.value for neuron in hidden_layer_1])

        output2  = self.chain[1](np.dot(neurons2, weights2) + bias2)

        # flatten the output
        output2 = output2.flatten()
        # update the value of the neurons
        for i in range(len(hidden_layer_2)):
            hidden_layer_2[i].value = output2[i]

        weights3 = []
        bias3 = []
        # get the weights
        for i in range(len(hidden_layer_2) * len(output_layer)):
            # get the weight
            weights3.append(self.weights[i + len(input_neurons) * len(hidden_layer_1) + len(hidden_layer_1) * len(hidden_layer_2)].value)
            # get the bias
            bias3.append(self.weights[i + len(input_neurons) * len(hidden_layer_1) + len(hidden_layer_1) * len(hidden_layer_2)].value)
        
        # reshape the weights
        weights3 = np.array(weights3).reshape(len(hidden_layer_2), len(output_layer))
        # reshape the bias
        bias3 = np.array(bias3).reshape(len(hidden_layer_2), len(output_layer))

        # get input_neurons as numpy array
        neurons3 = np.array([neuron.value for neuron in hidden_layer_2])

        output3  = self.chain[2](np.dot(neurons3, weights3) + bias3)

        # flatten the output
        output3 = output3.flatten()

        # update the value of the neurons
        for i in range(len(output_layer)):
            output_layer[i].value = output3[i]

        # backpropagation
        # calculate the error
        # get the output neurons
        output_neurons = self.layers[-1].neurons
        # get the target
        target = 30
        learning_rate = 0.1
        # calculate the error
        error = target - output_neurons[0].value
        # calculate the derivative of the error
        d_error = error * chain_deriv([self.chain[-1]], output_neurons[0].value)
        # calculate the derivative of the output
        d_output = chain_deriv([self.chain[-1]], output_neurons[0].value)
        # calculate the derivative of the weights
        d_weights = np.dot(neurons3.T, d_error * d_output)
        # calculate the derivative of the bias
        d_bias = d_error * d_output
        # update the weights
        for i in range(len(hidden_layer_2) * len(output_layer)):
            self.weights[i + len(input_neurons) * len(hidden_layer_1) + len(hidden_layer_1) * len(hidden_layer_2)].value -= learning_rate * d_weights[i]

        # update the bias
        for i in range(len(output_layer)):
            self.weights[i + len(input_neurons) * len(hidden_layer_1) + len(hidden_layer_1) * len(hidden_layer_2)].bias -= learning_rate * d_bias

        # calculate the derivative of the error
        d_error = np.dot(d_error * d_output, weights3.T)
        # calculate the derivative of the output
        d_output = deriv(self.chain[-2], neurons3)
        # calculate the derivative of the weights
        d_weights = np.dot(np.reshape(neurons2, (10, -1)), d_error * d_output)
        # calculate the derivative of the bias
        d_bias = d_error * d_output

        # flatten the d_weights
        d_weights = d_weights.flatten()
        # update the weights
        for i in range(len(hidden_layer_1) * len(hidden_layer_2)):
            self.weights[i + len(input_neurons) * len(hidden_layer_1)].value -= learning_rate * d_weights[i]

        # update the bias
        for i in range(len(hidden_layer_2)):
            self.weights[i + len(input_neurons) * len(hidden_layer_1)].bias -= learning_rate * d_bias

        # calculate the derivative of the error
        d_error = np.dot(d_error * d_output, weights2.T)
        # calculate the derivative of the output
        d_output = deriv(self.chain[-3], neurons2)
        # calculate the derivative of the weights
        d_weights = np.dot(np.reshape(neurons1, (6, -1)), d_error * d_output)
        # calculate the derivative of the bias
        d_bias = d_error * d_output

        # flatten the weights
        d_weights = d_weights.flatten()
        # update the weights
        for i in range(len(input_neurons) * len(hidden_layer_1)):
            self.weights[i].value -= learning_rate * d_weights[i]

        # update the bias
        for i in range(len(hidden_layer_1)):
            self.weights[i].bias -= learning_rate * d_bias


        
    def activation_function(self, value):
        # apply the activation function
        # relu function
        if value > 0:
            return value
        else:
            return 0

class Weight:
    def __init__(self, neuron1, neuron2, value):
        self.value = value
        self.neuron1 = neuron1
        self.neuron2 = neuron2
        self.bias = 0.001

    def draw(self, screen):
        #draw line between two neurons
        pygame.draw.line(screen, (0, 0, 0), (self.neuron1.x, self.neuron1.y), (self.neuron2.x, self.neuron2.y), 1)

    def update(self):
        pass

    def __str__(self) -> str:
        return str(self.value)


class Visulize:
    def __init__(self):
        self.scene = []
        self.setting = Setting()
        self.screen = pygame.display.set_mode((self.setting.screen_width, self.setting.screen_height))

        self.awake()
    
    def awake(self):
        # self.scene.append(Network(100, 530, 4, 15, (0, 0, 255),20, 10, 5, 3))
        # self.scene.append(Network(100, 530, 3, 15, (0, 0, 255),4, 3, 2))
        # self.scene.append(Network(100, 530, 3, 2, (0, 0, 255),784, 128, 64, 10))
        # self.scene.append(Network(100, 530, 4, 18, (0, 0, 255),4, 10, 10, 1))
        self.scene.append(Network(100, 530, 4, 18, (0, 0, 255), [sigmoid, relu, leacky_relu], 6, 10, 10, 1))

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