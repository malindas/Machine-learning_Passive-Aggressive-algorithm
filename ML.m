clc; % clear the command Window
clear all; % Clear all Workspace
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%     Author : Malinda Sulochana Silva
%     Dept. of Electrical and Electronic Engineering
%     Faculty of Engingeering
%     University of Peradeniya.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load the dataset 
    dataset = load('data.csv');

% Separate Input from the dataset
% Let 'x' be the input dataset
    x = dataset(:,2:10);
    
% Normalize the dataset
    x = x ./ 10;
    
% Separate output
% Let 'y' be the output 
    y = dataset(:,11);
    
% Adjust output as (-1,+1) for which given as (2,4)
    for i =1:length(y)
        if y(i) == 2 
            y(i) = -1;
        elseif y(i) == 4
            y(i) = 1;
        end
    end

len = length(y); % length of the dataset;

% Separating Testing and Training data out of the dataset
% Training dataset (2/3) of the entire dataset
    xTrain = x(1:len*(2/3),:);
    yTrain = y(1:len*(2/3));
% Test dataset (1/3) of the entire dataset
    xTest = x(len*(2/3):len,:);
    yTest = y(len*(2/3):len);
    
%==========================================================================
% _____________________________Training Data_______________________________



% Let aggressive parameter be C
    C=5;
% Initialize weight Matrix
    W = (zeros(1,9));

    % Let 't' be the iterations (t = 1,2,10)
for t = 1:3
    %prediction carriyng out for entire training dataset
    for i = 1:length(yTrain)
    % prediction Phase
        Y_predict = sign(W * (xTrain(i,:)' ));
     
        
    % Calculating the hinge loss
            if Y_predict >= 1
                hinge_loss = 0; % No loss
            else                
                hinge_loss = 1 - yTrain(i); % Loss
            end
   
      
    % Calculating the Suffer Loss
    suffer_loss = max([0,  1 - ( yTrain(i) *(W * xTrain(i,:)' ))]);
    
%++++++++++++++++++++++++++++ Update Phase+++++++++++++++++++++++++++++++++     

     %(Lagrange Multiplier)
        % (PA)   
            T1 = (suffer_loss) / norm(xTrain(i,:), 2);
        % (PA-I)
            T2 = min([C,T1]);
        % (PA-II)
            T3 = (suffer_loss) / (norm(xTrain(i,:), 2)+(1/2*C));
            
     % weights
            YX_Train =  (yTrain(i) .* xTrain(i,:));
            W = W + (T1 .* YX_Train) ;
    end       
end

%_________________________Done adjusting Weight____________________________
%________________________Training Data Complete_____________________________

%==========================================================================
%_____________________________Testing Data_________________________________

    prediction = sign(W * xTest')';   

%==========================================================================
%____________________________ Testing Accuracy ____________________________    
    deviation = prediction - yTest;
    
    Accurate_predictions=0;
    for i = 1:length(prediction)
        if deviation(i) == 0
            Accurate_predictions = Accurate_predictions + 1;
        end
    end
    
    Accuracy = Accurate_predictions / length(prediction)*100;
    fprintf('Accuracy = %f %% \n',Accuracy);
    fprintf('Weight Matrix = ');disp(W);
    

%_______________Copyright. All Rights Reserved_____________________________