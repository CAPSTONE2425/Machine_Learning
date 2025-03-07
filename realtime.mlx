clear; clc;


load('random_forest_model.mat', 'random_forest_model');
Fs = 1000;  % Sampling rate
window_size = 250;  % Window size (adjustable)
overlap = 0.9;  % Overlap percentage
step_size = round(window_size * (1 - overlap)); % Compute step size

disp(" Real-Time EMG Processing Started...");

time_limit = 60; % Run for 60 seconds
tic; % Start timer

while toc < time_limit
    % Simulated real-time EMG input (Replace with actual live sensor data)
    emg_signal = randn(window_size, 2); % Simulated 2-channel EMG
    
    filtered_fingers = filter_emg(emg_signal(:,1), Fs);
    filtered_thumb = filter_emg(emg_signal(:,2), Fs);
    features_fingers = extract_features(filtered_fingers);
    features_thumb = extract_features(filtered_thumb);
    features = [features_fingers, features_thumb];
    predicted_label = str2double(predict(random_forest_model, features));
    
    % Display Prediction
    if predicted_label == 1
        disp("🖐 Open Hand Detected");
    else
        disp("✊ Closed Hand Detected");
    end
    
    % Simulated delay (Replace with real-time sampling rate handling)
    pause(0.1); % 100ms delay to simulate real-time processing
end

disp("✅ Real-Time Processing Completed.");

