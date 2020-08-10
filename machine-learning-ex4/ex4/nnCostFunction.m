function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1)); 
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1)); 

% Substituindo X e y para somente algumas amostras para facilitar o debug
%X = X([1, 2500, 5000], :);
%y = y([1, 2500, 5000]);

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

  % Parte 1 - Calculando Cost Function J sem regularização 
  % size(Theta1)
  % size(Theta2)
  
  yVectorized = zeros(m, num_labels);

  for i=1:num_labels
    yVectorized(:,i) = (y == i);
  end

  % Adicionando a coluna referente ao "bias unit" nos dados de treino"
  a1 = [ones(size(X, 1), 1), X];
 
  z2 = a1 * Theta1';

  a2 = sigmoid(z2);
  a2 = [ones(size(a2, 1), 1), a2];

  z3 = a2 * Theta2';

  h = sigmoid(z3);

  J = 1/m * sum(sum((- yVectorized .* log(h)) - ((-yVectorized + 1) .* log(-h  + 1))));

  % Parte 2 - Incluindo regularização na Cost Function J
  % Para usar as matrizes Theta1 e Theta2, deve-se excluir a primeira colunar
  % referente aos parâmetros do bias unit

  reg = (lambda/(2*m)) * (sum(sum (Theta1(:, 2:end).^ 2)) + sum(sum (Theta2(:, 2:end).^ 2)));

  reg
  J = J + reg;

  % Backpropagation Algorithm
  for t = 1:m 
    X_t = X(t, :)';
    y_t = yVectorized(t, :)';

    % Inputs (Primeira camada da Rede)
    a1_t = [1; X_t];
    z2_t = Theta1 * a1_t;

    % Segunda camada da Rede
    a2_t = sigmoid(z2_t);
    a2_t = [1; a2_t];

    % Outputs (Terceira camada da Rede)
    z3_t = Theta2 * a2_t; 
    a3_t = sigmoid(z3_t);

    delta_3 = a3_t - y_t;
    delta_2 = (Theta2' * delta_3) .* [1; sigmoidGradient(z2_t)];
    delta_2 = delta_2(2:end);

%    delta_2 = Theta2' * delta_3;
%    delta_2 = delta_2(2:end) .* sigmoidGradient(z2_t);
    
    Theta2_grad = Theta2_grad + delta_3 * a2_t';
    Theta1_grad = Theta1_grad + delta_2 * a1_t';
  end

  Theta1_grad = Theta1_grad * (1/m);
  Theta2_grad = Theta2_grad * (1/m);

%  size(Theta1_grad)
%  size(Theta2_grad)
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
