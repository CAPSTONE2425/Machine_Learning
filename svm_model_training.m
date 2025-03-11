clear; clc;

% Define the folder where EMG files are stored
folder_path = "C:\Users\irede\Downloads\capstone";

% Check if the folder exists
if ~isfolder(folder_path)
    error("Error: The folder path %s does not exist!", folder_path);
end

% Get list of open and closed EMG data files
open_files = dir(fullfile(folder_path, '**', 'data_Open_*.txt'));
closed_files = dir(fullfile(folder_path, '**', 'data_Closed_*.txt'));

% Check if dataset contains both open and closed hand files
disp("Total Open Hand Files:"), disp(length(open_files));
disp("Total Closed Hand Files:"), disp(length(closed_files));

if isempty(open_files) || isempty(closed_files)
    error("🚨 Error: Missing open or closed hand files. Check dataset!");
end

% Initialize arrays
features_fingers = [];
features_thumb = [];
labels = [];

for i = 1:length(open_files)
    % Read EMG Data
    open_data = readmatrix(fullfile(open_files(i).folder, open_files(i).name));
    closed_data = readmatrix(fullfile(closed_files(i).folder, closed_files(i).name));

    % Ensure correct dimensions
    if size(open_data, 2) < 2 || size(closed_data, 2) < 2
        error("🚨 Error: %s or %s does not contain two columns.", open_files(i).name, closed_files(i).name);
    end

    % Process Data
    open_fingers = filter_emg(open_data(:,1));
    open_thumb = filter_emg(open_data(:,2));
    closed_fingers = filter_emg(closed_data(:,1));
    closed_thumb = filter_emg(closed_data(:,2));

    % Ensure data is valid before extracting features
    if isempty(open_fingers) || isempty(closed_fingers)
        error("🚨 Error: One of the data sets is empty. Check file contents!");
    end

    % Extract Features for Each Sample
    open_features_fingers = cell2mat(arrayfun(@(idx) extract_features(open_fingers(idx,:)), 1:size(open_fingers,1), 'UniformOutput', false)');
    closed_features_fingers = cell2mat(arrayfun(@(idx) extract_features(closed_fingers(idx,:)), 1:size(closed_fingers,1), 'UniformOutput', false)');

    open_features_thumb = cell2mat(arrayfun(@(idx) extract_features(open_thumb(idx,:)), 1:size(open_thumb,1), 'UniformOutput', false)');
    closed_features_thumb = cell2mat(arrayfun(@(idx) extract_features(closed_thumb(idx,:)), 1:size(closed_thumb,1), 'UniformOutput', false)');

    % Ensure Feature Dimensions Are Consistent
    feature_dim = size(open_features_fingers, 2);
    if any([size(closed_features_fingers, 2), size(open_features_thumb, 2), size(closed_features_thumb, 2)] ~= feature_dim)
        error("🚨 Error: Feature dimensions do not match! Check extract_features()");
    end

    % Append extracted features (only once)
    features_fingers = [features_fingers; open_features_fingers; closed_features_fingers];
    features_thumb = [features_thumb; open_features_thumb; closed_features_thumb];

    % Assign Labels: 1 = open hand, 0 = closed hand
    num_open_samples = size(open_features_fingers,1);
    num_closed_samples = size(closed_features_fingers,1);
    new_labels = [ones(num_open_samples,1); zeros(num_closed_samples,1)];
    
    % Append labels
    labels = [labels; new_labels];

end

% Ensure labels and features are properly aligned
if size(features_fingers,1) ~= size(labels,1)
    error("🚨 Error: Features (%d) and labels (%d) do not match!", size(features_fingers,1), size(labels,1));
end

% Balance dataset if necessary
num_ones = sum(labels == 1);
num_zeros = sum(labels == 0);

if num_ones > num_zeros
    num_missing = num_ones - num_zeros;
    valid_idx = find(labels == 0);
    num_valid_samples = length(valid_idx);

    if num_valid_samples == 0
        error("🚨 Error: No valid closed-hand samples found to balance dataset.");
    end

    num_selected = min(num_missing, num_valid_samples);

    if num_selected > 0
        extra_features = features_fingers(valid_idx(1:num_selected), :);
        extra_thumb_features = features_thumb(valid_idx(1:num_selected), :);
        extra_labels = zeros(num_selected, 1);

        features_fingers = [features_fingers; extra_features];
        features_thumb = [features_thumb; extra_thumb_features];
        labels = [labels; extra_labels];
    end
end

% Ensure final dataset consistency
min_samples = min(size(features_fingers,1), size(labels,1));
features_fingers = features_fingers(1:min_samples, :);
features_thumb = features_thumb(1:min_samples, :);
labels = labels(1:min_samples, :);

disp("Final Sizes Before Training:");
disp("Features (Fingers):"), disp(size(features_fingers));
disp("Features (Thumb):"), disp(size(features_thumb));
disp("Labels:"), disp(size(labels));

if size(features_fingers,1) ~= size(labels,1)
    error("🚨 Error: Features and labels do not match in size!");
end

%%%%%%%%%
% Reduce sample size for faster training
sample_size = min(2000, size(features_fingers,1)); % Use 10,000 samples or all available

% Select a random subset of data
random_indices = randperm(size(features_fingers,1), sample_size);

% Reduce dataset size
features_fingers_sample = features_fingers(random_indices, :);
features_thumb_sample = features_thumb(random_indices, :);
labels_sample = labels(random_indices, :);

disp("Reduced Dataset Sizes Before Training:");
disp("Features (Fingers):"), disp(size(features_fingers_sample));
disp("Features (Thumb):"), disp(size(features_thumb_sample));
disp("Labels:"), disp(size(labels_sample));


% Train SVM for fingers
%opt = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',cvpartition(labels, 'KFold', 10));
opt = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',cvpartition(sample_size, 'KFold', 5));



%fingers_model = fitcsvm(features_fingers, labels, 'KernelFunction', 'rbf', ...
fingers_model = fitcsvm(features_fingers_sample, labels_sample, 'KernelFunction', 'rbf', ...
'BoxConstraint', 10, 'OptimizeHyperparameters', {'KernelScale', 'BoxConstraint'}, ...
    'HyperparameterOptimizationOptions', opt);

thumb_model = fitcsvm(features_thumb_sample, labels_sample, 'KernelFunction', 'rbf', ...
    'BoxConstraint', 10, 'OptimizeHyperparameters', {'KernelScale', 'BoxConstraint'}, ...
    'HyperparameterOptimizationOptions', opt);

cv_fingers = crossval(fingers_model, 'KFold', 10);
cv_thumb = crossval(thumb_model, 'KFold', 10);

fprintf("Fingers Model 10-Fold CV Error: %.2f%%\n", kfoldLoss(cv_fingers) * 100);
fprintf("Thumb Model 10-Fold CV Error: %.2f%%\n", kfoldLoss(cv_thumb) * 100);

