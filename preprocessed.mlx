clear; clc;

load('trained_random_forest.mat', 'random_forest_model');

%  Load the sample to check
random_sample_file = "C:\Users\irede\Downloads\data_tireni_2";
random_sample = readmatrix(random_sample_file);

if size(random_sample, 2) < 2
    error(" Error: The input sample file does not have the expected 2 EMG channels.");
end

fingers_emg = random_sample(:,1);
thumb_emg = random_sample(:,2);
Fs = 1000;

% Filter EMG Signals
filtered_fingers = filter_emg(fingers_emg, Fs);
filtered_thumb = filter_emg(thumb_emg, Fs);

% Extract Features
features_fingers = extract_features(filtered_fingers);
features_thumb = extract_features(filtered_thumb);

% Combine Features
test_features = [features_fingers, features_thumb];

% Predict Open or Closed Hand
test_features = reshape(test_features, 1, []);
predicted_label = str2double(predict(random_forest_model, test_features));

% Display Result
if predicted_label == 1
    disp("🖐️ Prediction: Open Hand");
else
    disp("✊ Prediction: Closed Hand");
end

