clear; clc;

%load emg data
folder_path = "C:\Users\irede\Downloads\capstone";

% Check if folder exists
if ~isfolder(folder_path)
    error("🚨 Error: The folder path %s does not exist!", folder_path);
end


open_files = dir(fullfile(folder_path, '**', 'data_Open_*.txt'));
closed_files = dir(fullfile(folder_path, '**', 'data_Closed_*.txt'));

disp("Total Open Hand Files:"), disp(length(open_files));
disp("Total Closed Hand Files:"), disp(length(closed_files));

if isempty(open_files) || isempty(closed_files)
    error("🚨 Error: Missing open or closed hand files. Check dataset!");
end

data = [];
labels = [];
Fs = 1000;  % Sampling rate
window_size = 250;  % Window size (adjustable)
overlap = 0.75;  % Overlap percentage
step_size = round(window_size * (1 - overlap)); % Compute step size

for i = 1:length(open_files)
    % Read Open Hand Data
    open_data = readmatrix(fullfile(open_files(i).folder, open_files(i).name));
    closed_data = readmatrix(fullfile(closed_files(i).folder, closed_files(i).name));

    % Ensure correct dimensions
    if size(open_data, 2) < 2 || size(closed_data, 2) < 2
        error("🚨 Error: Incorrect file format.");
    end

    open_fingers = filter_emg(open_data(:,1), Fs);
    open_thumb = filter_emg(open_data(:,2), Fs);
    closed_fingers = filter_emg(closed_data(:,1), Fs);
    closed_thumb = filter_emg(closed_data(:,2), Fs);

    open_features = [];
    closed_features = [];

    % Process Open Hand Data
    for start_idx = 1:step_size:(length(open_fingers) - window_size)
        window_fingers = open_fingers(start_idx:start_idx + window_size - 1);
        window_thumb = open_thumb(start_idx:start_idx + window_size - 1);
        features_fingers = extract_features(window_fingers);
        features_thumb = extract_features(window_thumb);

        % Combine features
        open_features = [open_features; features_fingers, features_thumb];
    end

    % Process Closed Hand Data
    for start_idx = 1:step_size:(length(closed_fingers) - window_size)
        window_fingers = closed_fingers(start_idx:start_idx + window_size - 1);
        window_thumb = closed_thumb(start_idx:start_idx + window_size - 1);
        features_fingers = extract_features(window_fingers);
        features_thumb = extract_features(window_thumb);

        % Combine features
        closed_features = [closed_features; features_fingers, features_thumb];
    end

    % Validate extracted features
    if isempty(open_features) || isempty(closed_features)
        warning("⚠ Warning: No valid features extracted for file index %d. Skipping...", i);
        continue; % Skip to next iteration if no features extracted
    end

    % Append Data
    data = [data; open_features; closed_features];
    labels = [labels; ones(size(open_features,1),1); zeros(size(closed_features,1),1)];
end


disp("✅ Total Samples in Dataset: "), disp(size(data, 1));
csvwrite('training_samples.csv', data);
csvwrite('training_labels.csv', labels);
disp("✅ Training samples saved to 'training_samples.csv'");
disp("✅ Labels saved to 'training_labels.csv'");


k = 20; % k-fold cross-validation
cv = cvpartition(size(data, 1), 'KFold', k);
cv_accuracy = zeros(k, 1); % Store accuracy for each fold

disp("🔄 Performing K-Fold Cross-Validation...");

for fold = 1:k
    train_idx = training(cv, fold);
    test_idx = test(cv, fold);

    train_data = data(train_idx, :);
    train_labels = labels(train_idx);
    test_data = data(test_idx, :);
    test_labels = labels(test_idx);

    num_trees = 300;  % Number of decision trees
    random_forest_model = TreeBagger(num_trees, train_data, train_labels, ...
        'OOBPrediction', 'On', 'Method', 'classification');
    save('trained_random_forest.mat', 'random_forest_model');

    predicted_labels = str2double(predict(random_forest_model, test_data)); % Convert predicted labels to numeric
    accuracy = sum(predicted_labels == test_labels) / length(test_labels);
    cv_accuracy(fold) = accuracy;

    fprintf("Fold %d Accuracy: %.2f%%\n", fold, accuracy * 100);
end

% Compute and display final average accuracy
final_accuracy = mean(cv_accuracy) * 100;
fprintf("\n Final K-Fold Cross-Validation Accuracy: %.2f%%\n", final_accuracy);



