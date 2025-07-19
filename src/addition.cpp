#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>

//sigmoid activation function and derivative
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

//random weight between -1 and 1
double random_weight() {
    return ((double) rand() / RAND_MAX) * 2 - 1;
}

class Neuron {
public:
    std::vector<double> weights;
    double bias;
    double output;
    double delta;

    Neuron(int num_inputs) {
        for (int i = 0; i < num_inputs; ++i) {
            weights.push_back(random_weight());
        }
        bias = random_weight();
    }

    double activate(const std::vector<double>& inputs) {
        double net_input = bias;
        for (size_t i = 0; i < inputs.size(); ++i) {
            net_input += weights[i] * inputs[i];
        }
        output = sigmoid(net_input);
        return output;
    }

    void save(std::ofstream& out) {
        for (double w : weights)
            out << w << " ";
        out << bias << "\n";
    }

    void load(std::ifstream& in) {
        for (double& w : weights)
            in >> w;
        in >> bias;
    }
};

class NeuralNetwork {
public:
    std::vector<Neuron> hidden;
    Neuron output_neuron;

    NeuralNetwork(int input_size, int hidden_size)
        : output_neuron(hidden_size) {
        for (int i = 0; i < hidden_size; ++i) {
            hidden.emplace_back(input_size);
        }
    }

    double feedforward(const std::vector<double>& inputs) {
        std::vector<double> hidden_outputs;
        for (auto& neuron : hidden) {
            hidden_outputs.push_back(neuron.activate(inputs));
        }
        return output_neuron.activate(hidden_outputs);
    }

    void train(const std::vector<double>& inputs, double target, double learning_rate) {
        //forward pass
        double output = feedforward(inputs);
        //output error and delta
        double error = target - output;
        output_neuron.delta = error * sigmoid_derivative(output_neuron.output);
        //hidden layer delta
        for (size_t i = 0; i < hidden.size(); ++i) {
            hidden[i].delta = output_neuron.delta * output_neuron.weights[i] * sigmoid_derivative(hidden[i].output);
        }
        //update output weights
        for (size_t i = 0; i < output_neuron.weights.size(); ++i) {
            output_neuron.weights[i] += learning_rate * output_neuron.delta * hidden[i].output;
        }
        output_neuron.bias += learning_rate * output_neuron.delta;
        //update hidden weights
        for (auto& neuron : hidden) {
            for (size_t i = 0; i < neuron.weights.size(); ++i) {
                neuron.weights[i] += learning_rate * neuron.delta * inputs[i];
            }
            neuron.bias += learning_rate * neuron.delta;
        }
    }

    void save(const std::string& filename) {
        std::ofstream out(filename);
        for (auto& neuron : hidden)
            neuron.save(out);
        output_neuron.save(out);
        out.close();
    }

    void load(const std::string& filename) {
        std::ifstream in(filename);
        for (auto& neuron : hidden)
            neuron.load(in);
        output_neuron.load(in);
        in.close();
    }
};

int main() {
    srand(static_cast<unsigned>(time(0)));
     //2 inputs, 8 hidden neurons
    NeuralNetwork nn(2, 120); 
    double learning_rate = 0.01;
    std::cout << "Training...\n";
    for (int epoch = 0; epoch < 10000000; ++epoch) {
        double a = (rand() % 1000) / 100.0;
        double b = (rand() % 1000) / 100.0;
        double target = (a + b) / 20.0;
        nn.train({a / 10.0, b / 10.0}, target, learning_rate);
    }
    nn.save("trained_network.txt");
    std::cout << "Model saved to 'trained_network.txt'.\n";
    //loads model
    NeuralNetwork nn_loaded(2, 120);
    nn_loaded.load("trained_network.txt");
    std::cout << "\nTesting loaded network:\n";
    for (double a = 0; a <= 10; a += 2) {
        for (double b = 0; b <= 10; b += 2) {
            double predicted = nn_loaded.feedforward({a / 10.0, b / 10.0}) * 20.0;
            std::cout << a << " + " << b << " = " << predicted << " (expected " << (a + b) << ")\n";
        }
    }
    return 0;
}
