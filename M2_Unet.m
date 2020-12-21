%% M2.m file
%
% %
% This file is for your implementation and validation of M2 (sections 4 and 5).
% It contains useful functions (sections 1 and 2) as well as a partial
% implementation of the U-net for you to finalise (section 3).

%% 2. Normalization of the training and testing datasets
% Working with a normalized data set can help enhance the training of
% networks. The function below will be of help.

% Training images - datastore for dirty and true images
ds_trainingset_dirty = imageDatastore('..\training_set\trainingset_dirty\*.png');   % can add any processing function here
ds_trainingset_true = imageDatastore('..\training_set\trainingset_true\*.png');
% combine into a single datastore that feeds data to trainNetwork
ds = combine(ds_trainingset_dirty,ds_trainingset_true);
ds = transform(ds,@commonPreprocessing);    % apply preprocessing
ds = shuffle(ds);   % randomly shuffle the data

% split into training and validation sets
N = 7115;
p = 0.95;        % 70% used for training, 30% for validation
train_inds = 1:floor(p*N);
val_inds = floor(p*N)+1:N;
ds_train = subset(ds,train_inds);
ds_val = subset(ds,val_inds);

%% 3. U-net architecture and implementation (to be finalised) 
% See project document Appendix A for architecture details 

input_size = 320;
num_channels = 64;

layers = [
    imageInputLayer([input_size input_size 1], 'Normalization','none','Name','Input')
    
    % Depth 1
    convolution2dLayer(3,num_channel,'Padding','same','Name','conv11')
    batchNormalizationLayer('Name','bn11')
    reluLayer('Name','relu11')

    convolution2dLayer(3,num_channel,'Padding','same','Name','conv12')
    batchNormalizationLayer('Name','bn12')
    reluLayer('Name','relu12')
    
    convolution2dLayer(3,num_channel,'Padding','same','Name','conv13')
    batchNormalizationLayer('Name','bn13')
    reluLayer('Name','relu13')
    
    maxPooling2dLayer(2,'Stride',2,'Name','MP1')
    
    % Depth 2
    convolution2dLayer(3,num_channel*2,'Padding','same','Name','conv21')
    batchNormalizationLayer('Name','bn21')
    reluLayer('Name','relu21')
    
    convolution2dLayer(3,num_channel*2,'Padding','same','Name','conv22')
    batchNormalizationLayer('Name','bn22')
    reluLayer('Name','relu22')
    
    maxPooling2dLayer(2,'Stride',2, 'Name','MP2')
    
    % Depth 3
    convolution2dLayer(3,num_channel*4,'Padding','same','Name','conv31')
    batchNormalizationLayer('Name','bn31')
    reluLayer('Name','relu31')
    
    convolution2dLayer(3,num_channel*4,'Padding','same','Name','conv32')
    batchNormalizationLayer('Name','bn32')
    reluLayer('Name','relu32')
    
    transposedConv2dLayer(3,num_channel*2,'Cropping','same','Stride',2,'Name','up32')
    batchNormalizationLayer('Name','bn33')
    reluLayer('Name','relu33')
     
    % Back to depth 2
    depthConcatenationLayer(2,'Name','concat2')
    convolution2dLayer(3,num_channel*2,'Padding','same','Name','conv23')
    batchNormalizationLayer('Name','bn23')
    reluLayer('Name','relu23')
    
    convolution2dLayer(3,num_channel*2,'Padding','same','Name','conv24')
    batchNormalizationLayer('Name','bn24')
    reluLayer('Name','relu24')
    
    transposedConv2dLayer(3,num_channel,'Cropping','same','Stride',2,'Name','up24')
    batchNormalizationLayer('Name','bn25')
    reluLayer('Name','relu25')
    
    % Back to depth 1
    depthConcatenationLayer(2,'Name','concat1')
    convolution2dLayer(3,num_channel,'Padding','same','Name','conv14')
    batchNormalizationLayer('Name','bn14')
    reluLayer('Name','relu14')
    
    convolution2dLayer(3,num_channel,'Padding','same','Name','conv15')
    batchNormalizationLayer('Name','bn15')
    reluLayer('Name','relu15')
    
    convolution2dLayer(3,1,'Padding','same','Name','conv16')
    batchNormalizationLayer('Name','bn16')
    reluLayer('Name','relu16')
    
    convolution2dLayer(1,1,'Padding','same','Name','conv17')
    
    additionLayer(2,'Name','add17')
    
    regressionLayer('Name','output')
];

% connect layers to complete the network
lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph,'relu22','concat2/in2');
lgraph = connectLayers(lgraph,'relu13','concat1/in2');
lgraph = connectLayers(lgraph,'Input','add17/in2');

% define training options
BatchSize = 8;
options = trainingOptions('adam', ...
    'MaxEpochs',30, ...
    'MiniBatchSize',BatchSize, ...             % try 5
    'InitialLearnRate',5e-3, ...     % 1e-4
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',10, ...       % drop the learning rate every 10 epochs
    'LearnRateDropFactor',0.1, ...      % drop the learning rate 10 times
    'ValidationData',ds_val, ...        % use ds_val as validation data
    'ValidationFrequency',ceil(floor(p*N)/BatchSize), ...        % roughly validate per epoch
    'Shuffle','never', ...
    'Plots','training-progress');

% uncomment the following line if you would like to retrain your network
% otherwise load in previously saved results
% [M2_net,info] = trainNetwork(ds_train, lgraph, options);
%% 5. M2 validation - on the validation set
N_val = 356;%length(val_inds);

% allocate space for arrays and cells
rsnr_val = zeros(N_val,2);
ssim_val = zeros(N_val,2);
img_cell = cell(N_val,3);

% loop through images in the validation set
for i = 1:N_val
    imgs = read(ds_val);
    true_img = imgs{2};
    bp_img = imgs{1};
    
    m2_img = double(predict(M2_net,bp_img));    % predicted image
    imgs{3} = m2_img;                           % save predicted image in the third column
    img_cell(i,1:3) = imgs;
    
    rsnr_val(i,1) = 20*log10(norm(true_img(:))/norm(true_img(:)-bp_img(:)));     % rsnr of dirty image
    rsnr_val(i,2) = 20*log10(norm(true_img(:))/norm(true_img(:)-m2_img(:)));     % rsnr of predicted image
    ssim_val(i,1) = ssim(true_img,bp_img);                          % ssim of dirty image
    ssim_val(i,2) = ssim(true_img,m2_img);                          % ssim of predicted image
end

% plot a result
n = randi(N_val);    % random integer between 1 and 20
figure;
montage({img_cell{n,2},img_cell{n,1},img_cell{n,3}},'Size',[1 3]);   % put noisy, true and predicted images together
title('True (Left), Dirty(Middle) and Predicted (Right)')

% show mean rsnr and ssim
disp('----- Validation results ------')
disp(['mean rsnr of dirty images is ', num2str(mean(rsnr_val(:,1)))])
disp(['mean rsnr of predicted images is ', num2str(mean(rsnr_val(:,2)))])
disp(['mean ssim of dirty images is ', num2str(mean(ssim_val(:,1)))])
disp(['mean ssim of predicted images is ', num2str(mean(ssim_val(:,2)))])

% show std rsnr and ssim
disp('----- Validation results ------')
disp(['std rsnr of dirty images is ', num2str(std(rsnr_val(:,1)))])
disp(['std rsnr of predicted images is ', num2str(std(rsnr_val(:,2)))])
disp(['std ssim of dirty images is ', num2str(std(ssim_val(:,1)))])
disp(['std ssim of predicted images is ', num2str(std(ssim_val(:,2)))])

%% normalise
% The commonPreprocessing helper function defines the preprocessing that is
% common to the training, validation, and test sets. This functions uses the
% im2double function converts image to double precision (between [0,1]) 
% The helper function requires the format of the input data to be a
% two-column cell array of image data, which matches the format of data
% returned by the read function of CombinedDatastore.
function dataOut = commonPreprocessing(data)
    dataOut = cell(size(data));
    for col = 1:size(data,2)
        for idx = 1:size(data,1)
            temp = data{idx,col};
            temp = im2double(temp);   % im2double or im2single
            dataOut{idx,col} = temp;
        end
    end
end