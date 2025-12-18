import json

def example_weights():
    """This is an example function that returns weights.
    Use this function as a template for optimal_step_weights and optimal_sigmoid_weights.
    You do not need to modify this class for this assignment.
    """
    w = {}

    w['hidden_layer_0_1'] = 0
    w['hidden_layer_1_1'] = 0
    w['hidden_layer_2_1'] = 0
    w['hidden_layer_0_2'] = 0
    w['hidden_layer_1_2'] = 0
    w['hidden_layer_2_2'] = 0
    w['hidden_layer_0_3'] = 0
    w['hidden_layer_1_3'] = 0
    w['hidden_layer_2_3'] = 0

    w['output_layer_0'] = 0
    w['output_layer_1'] = 0
    w['output_layer_2'] = 0
    w['output_layer_3'] = 0

    return w


def optimal_step_weights():
    """Return the optimal weights for the neural network with a step activation function.
    
    This function will not be graded if there are no optimal weights.
    See the PDF for instructions on what each weight represents.
    
    The hidden layer weights are notated by [1] on the problem set and 
    the output layer weights are notated by [2].

    This function should return a dict with elements for each weight, see example_weights above.

    """
    w = example_weights()

    # *** START CODE HERE ***
    # Strategy: Create 3 decision boundaries that form a triangle
    # Triangle vertices: (0,0), (4,0), (2,3)
    
    # Hidden unit 1: y ≥ 0 (above bottom edge)
    w['hidden_layer_0_1'] = 0.1    # small positive bias
    w['hidden_layer_1_1'] = 0      # no x₁ term
    w['hidden_layer_2_1'] = 1      # positive y term
    
    # Hidden unit 2: -1.5x₁ - y + 6 ≥ 0 (below right edge)
    w['hidden_layer_0_2'] = 6      # bias
    w['hidden_layer_1_2'] = -1.5   # x₁ coefficient  
    w['hidden_layer_2_2'] = -1     # x₂ coefficient
    
    # Hidden unit 3: 1.5x₁ - y ≥ 0 (below left edge)
    w['hidden_layer_0_3'] = 0      # bias
    w['hidden_layer_1_3'] = 1.5    # x₁ coefficient
    w['hidden_layer_2_3'] = -1     # x₂ coefficient
    
    # Output layer: Inside triangle when ALL conditions met
    w['output_layer_0'] = 2.5      # high bias
    w['output_layer_1'] = -1       # reduce when h₁ active
    w['output_layer_2'] = -1       # reduce when h₂ active
    w['output_layer_3'] = -1       # reduce when h₃ active
    # *** END CODE HERE ***

    return w

def optimal_linear_weights():
    """Return the optimal weights for the neural network with a linear activation function for the hidden units.
    
    This function will not be graded if there are no optimal weights.
    See the PDF for instructions on what each weight represents.
    
    The hidden layer weights are notated by [1] on the problem set and 
    the output layer weights are notated by [2].

    This function should return a dict with elements for each weight, see example_weights above.

    """
    w = example_weights()

    # *** START CODE HERE ***
    # Note: Linear activation cannot achieve 100% accuracy on non-linearly separable data
    # This is the best linear approximation we can find
    w['hidden_layer_0_1'] = 1     
    w['hidden_layer_1_1'] = -1     
    w['hidden_layer_2_1'] = -1     
    
    w['hidden_layer_0_2'] = 0      
    w['hidden_layer_1_2'] = 1      
    w['hidden_layer_2_2'] = 0      
    
    w['hidden_layer_0_3'] = 0      
    w['hidden_layer_1_3'] = 0      
    w['hidden_layer_2_3'] = 1      
    
    # Output layer weights
    w['output_layer_0'] = 0
    w['output_layer_1'] = 1
    w['output_layer_2'] = -0.5
    w['output_layer_3'] = -0.5
    # *** END CODE HERE ***

    return w

if __name__ == "__main__":
    step_weights = optimal_step_weights()

    with open('output/step_weights', 'w') as f:
        json.dump(step_weights, f)

    linear_weights = optimal_linear_weights()

    with open('output/linear_weights', 'w') as f:
        json.dump(linear_weights, f)