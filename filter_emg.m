function filtered_signal = filter_emg(signal, Fs)
    % Ensure correct number of input arguments
    if nargin < 2
        error(" Error: Missing input arguments! Ensure both 'signal' and 'Fs' are passed.");
    end

    signal = signal(:);
    signal(isnan(signal) | isinf(signal)) = 0; 
    signal = signal - mean(signal);

    % Filter
    low_cutoff = 20;    % High-pass filter cutoff 
    high_cutoff = 450;  % Low-pass filter cutoff 
    notch_freq = 50;    % Notch filter

    % Ensure Fs is reasonable
    if Fs <= 0
        error(" Error: Invalid sampling frequency (Fs). Must be positive.");
    end

%High pass
    [b_high, a_high] = butter(4, low_cutoff / (Fs/2), 'high');
    signal = filtfilt(b_high, a_high, signal);
%Low-Pass 
    [b_low, a_low] = butter(4, high_cutoff / (Fs/2), 'low');
    signal = filtfilt(b_low, a_low, signal);
% Notch Filter 

    Wo = notch_freq / (Fs/2); % Normalize frequency
    BW = Wo / 35;  % Bandwidth for notch filter
    [b_notch, a_notch] = iirnotch(Wo, BW);
    filtered_signal = filtfilt(b_notch, a_notch, signal);

end


