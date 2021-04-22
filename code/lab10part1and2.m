clc
clear
patientA = load('Patient_A.mat').data;
patientB = load('Patient_B.mat').data;

patientA_test = load('training_seizure_A_condensed.mat').detection;
patientB_test = load('training_seizure_B_condensed.mat').detection;

% SD: 1-30 Hz
% up to 3 minutes
% slowing down stands out
% sampling rate = 3 kHz (3000 samples/sec)

sampling_rate = 3000;
window_seconds = 10;
window_samples = sampling_rate*window_seconds;
windows_per_signal = size(patientA, 2)/window_samples;

%THRESHOLD = [0.4 0.7 1.1 1.35];
%analyzeAndPlot(patientA, window_samples, windows_per_signal, THRESHOLD);
%patientB_prediction = analyzeAndPlot(patientB, window_samples, windows_per_signal, THRESHOLD, "Features");
%[features_overall, features_FP, features_TN] = preformanceMetrics(patientB_prediction, patientB_test);
%ch1 = trainingResultA(1,:);
%figure;
%plot(trainingResultA(1,:));
%patientB_prediction_LDA = my_LDA(patientA, patientB, patientA_test, window_samples, windows_per_signal);
%[LDA_overall, LDA_FP, LDA_TN] = preformanceMetrics(patientB_prediction_LDA, patientB_test);

%------------------ START HFO ----------------------------------%

% Berger bands      1-70	
% Fast gamma        70-150	
% Ripples           150-250	
% Fast Ripples      250-600	
% Units             1000-2000	

% HFO: 100-500 Hz
% 20-150 ms stands out

% Patient A, seizure can be seen on channel 3, sample 2480000
% Patient B, seizure can be seen on channel 2, sample 2800000

patientA_filtered_HFO = FilterHFO(patientA);
patientB_filtered_HFO = FilterHFO(patientB);
THRESHOLD = [0.76 0.84 1.1 1.35];
%analyzeAndPlot(patientA_filtered_HFO, window_samples, windows_per_signal, THRESHOLD);
patientB_prediction_HFO = analyzeAndPlot(patientB_filtered_HFO, window_samples, windows_per_signal, THRESHOLD, "HFO");
[HFO_overall, HFO_FP, HFO_TN] = preformanceMetrics(patientB_prediction_HFO, patientB_test);

%------------------ END HFO ----------------------------------%

% for now output = # zero crossings per window
% have to use the normalization
function [numZeroCross] = zeroCrossings(signal, window_samples, windows_per_signal)
    zcd = dsp.ZeroCrossingDetector;
    numZeroCrossAll = double(zcd(signal'));
    AvgNumZeroCrossPerWindow = numZeroCrossAll/windows_per_signal;
    release(zcd);
    numZeroCross = [];
    for i=1:window_samples:length(signal)-window_samples
        smoothed_signal = smooth(signal(:,i:i+window_samples));
        x = double(zcd(smoothed_signal));
        x = x/AvgNumZeroCrossPerWindow;
        numZeroCross = [numZeroCross x];
    end
end

function [binary_result] = zeroCrossingsDetection(signal, window_samples, windows_per_signal, THRESHOLD)
    numZeroCross = zeroCrossings(signal, window_samples, windows_per_signal);
    %LOW_THRESHOLD = 0.55; %randomly chosen
    %HIGH_THRESHOLD = 0.95; %randomly chosen
    binary_result = [];
    for i=1:length(numZeroCross)
        if (numZeroCross(i) > THRESHOLD(1)) && (numZeroCross(i) < THRESHOLD(2))
            binary_result = [binary_result 1];
        else
            binary_result = [binary_result 0];
        end
    end
end

function [av_line_length] = averageLineLength(signal, windows_per_signal)
    total_line_length = sum(abs(diff(signal)));
    av_line_length = total_line_length/(windows_per_signal-1);
end

% normalize somehow
function [line_lengths] = lineLengths(signal, window_samples)
    line_lengths = [];
    for i=1:window_samples:length(signal)-window_samples
        differences = diff(signal(:,i:i+window_samples));
        line_length = sum(abs(differences));
        line_lengths = [line_lengths line_length];
    end
end

function [binary_result] = lineLengthsDetection(signal, window_samples, windows_per_signal, THRESHOLD)
    av_line_length = averageLineLength(signal, windows_per_signal);
    line_lengths = lineLengths(signal, window_samples);
    normalized_line_lengths = line_lengths/av_line_length;
    %THRESHOLD = 1.1; %randomly chosen
    binary_result = [];
    for i=1:length(normalized_line_lengths)
        if normalized_line_lengths(i) > THRESHOLD
            binary_result = [binary_result 1];
        else
            binary_result = [binary_result 0];
        end
    end
end

%do max minus min  in every bin instead?
%movmean command
function [signal_amps] = signalAmplitude(signal, window_samples, windows_per_signal)
    signal_amps = [];
    for i=1:window_samples:length(signal)-window_samples
        signal_amp = rms(signal(:,i:i+window_samples));
        signal_amps = [signal_amps signal_amp];
    end
end

function [binary_result] = signalAmplitudeDetection(signal, window_samples, windows_per_signal, THRESHOLD)
    average_amp = rms(signal);
    signal_amps = signalAmplitude(signal, window_samples, windows_per_signal);
    signal_amps_normalized = signal_amps/average_amp;
    %THRESHOLD = 1.35; %randomly chosen
    binary_result = [];
    for i=1:length(signal_amps_normalized)
        if signal_amps_normalized(i) > THRESHOLD
            binary_result = [binary_result 1];
        else
            binary_result = [binary_result 0];
        end
    end
end

function [return_all] = my_LDA(patientA, patientB, training_result, window_samples, windows_per_signal)
    return_all = [];
    f1 = figure;
    for k = 1:3
        testClass = [];
        j = 1;
        for i=1:window_samples:size(patientA,2)-window_samples
            sample = patientB(k,i:i+window_samples-1)';
            training = patientA(k,i:i+window_samples-1)';
            group = upsample(training_result(k,j), window_samples);
            x = classify(sample, training, group, 'linear')';
            testClass = [testClass x];
            j = j + 1;
        end
        
        subplot(3,1,k)
        hold on
        plot(patientB(k,:))
        plot(2000*testClass)
        xlim([0 5400000]);
        legend('signal', 'detection');
        hold off   

        testClass = downsample(testClass, window_samples);
        return_all = [return_all; testClass];
    end
    
    figure(f1);
    %ax = axes(f1);
    xlabel('Time bin (3 kHz)');
    ylabel('Voltage (uV)');
    %legend(ax, 'signal', 'detection');
    saveas_ = '../figures/lab10_LDA';
    savefig(append(saveas_, '.fig'))
    %saveas(f1, append(saveas_, '.jpg'))
end

function [returnVal] = analyzeAndPlot(signal, window_samples, windows_per_signal, THRESHOLD, lab)
      
    f1 = figure;
    xlim([0 5400000]);
    f2 = figure;
    xlim([0 5400000]);
    %subplot(4,1,1)
    returnVal = [];
    for i = 1:3
        %A3_zero_cross = upsample(1000*zeroCrossingsDetection(signal(i,:), window_samples, windows_per_signal), window_samples);
        %A3_line_lengths = upsample(1000*lineLengthsDetection(signal(i,:), window_samples, windows_per_signal), window_samples);
        %A3_signal_amps = upsample(1000*signalAmplitudeDetection(signal(i,:), window_samples, windows_per_signal), window_samples);
        A3_zero_cross = zeroCrossingsDetection(signal(i,:), window_samples, windows_per_signal, THRESHOLD(1:2));
        A3_line_lengths = lineLengthsDetection(signal(i,:), window_samples, windows_per_signal, THRESHOLD(3));
        A3_signal_amps = signalAmplitudeDetection(signal(i,:), window_samples, windows_per_signal, THRESHOLD(4));
        detection_prelim_prelim = (A3_zero_cross & A3_line_lengths & A3_signal_amps);
        detection_prelim = upsample(detection_prelim_prelim, window_samples);
        lim = max(signal(i,:));
        detection = lim*0.25*detection_prelim;
        returnVal = [returnVal; detection_prelim_prelim];
        figure(f1)
        subplot(3,1,i)
        hold on
        plot(signal(i,:))
        plot(detection)
        xlim([0 5400000]);
        legend('signal', 'detection');
        hold off
        
        A3_signal_amps = upsample(0.33*A3_signal_amps, window_samples);
        A3_line_lengths = upsample(0.66*A3_line_lengths, window_samples);
        A3_zero_cross = upsample(A3_zero_cross, window_samples);
        %A3_signal_amps(A3_signal_amps==0)=nan;
        %A3_line_lengths(A3_line_lengths==0)=nan;
        %A3_zero_cross(A3_zero_cross==0)=nan;
        figure(f2)
        subplot(3,1,i)
        hold on
        plot(A3_zero_cross)
        plot(A3_line_lengths)
        plot(A3_signal_amps)
        xlim([0 5400000]);
        legend('zero cross', 'line lengths', 'signal amps')
        hold off
             
    end
    
    figure(f1);
    %ax = axes(f1);
    xlabel('Time bin (3 kHz)');
    ylabel('Voltage (uV)');
    %legend(ax, 'signal', 'detection');
    saveas_ = '../figures/lab10_sig_' + lab;
    savefig(append(saveas_, '.fig'))
    saveas(f1, append(saveas_, '.jpg'))
    
    figure(f2);
    %ax = axes(f2);
    xlabel('Time bin (3 kHz)');
    ylabel('Detection (non-zero = detection)');
    %legend(ax, 'zero cross', 'line lengths', 'signal amps');
    saveas_ = '../figures/lab10_det_' + lab;
    savefig(append(saveas_, '.fig'))
    saveas(f2, append(saveas_, '.jpg'))
    
end

function [signals] = FilterHFO(signal)
    detected = false;
    signals = [];
    for i = 1:3
        [sig, filter] = bandpass(signal(i,:), [100 500], 3000);
        signal_filtered = filtfilt(filter, sig);
        signals = [signals; signal_filtered];
    end
    
end

function [] = plotStuffHFO(signal_raw, signal_filtered, channel, range)
    figure;
    subplot(2,1,1)
    hold on
    plot(signal_raw(channel,:));
    plot(signal_filtered);
    hold off
    subplot(2,1,2)
    hold on
    plot(signal_raw(channel,range));
    plot(signal_filtered(:,range));
    hold off
end

function [overall, FP, TN] = preformanceMetrics(prediction, test)
    performanceOverall = (test == prediction);
    total = size(performanceOverall, 1)*size(performanceOverall, 2);
    performanceSum = sum(performanceOverall, 'all');
    overall = performanceSum/total;

    falsePositive = [];
    trueNegative = [];
    for i=1:size(prediction,1)
        for j=1:size(prediction,2)
            falsePositive_ = ((prediction(i,j) == 1) && (test(i,j) == 0));
            falsePositive = [falsePositive falsePositive_];
            trueNegative_ = ((prediction(i,j) == 0) && (test(i,j) == 1));
            trueNegative = [trueNegative trueNegative_];
        end
    end
    performance_falsePositive = sum(falsePositive, 'all');
    performanceSum_trueNegative = sum(trueNegative, 'all');
    FP = performance_falsePositive/total;
    TN = performanceSum_trueNegative/total;
end

% don't think this is useful
function [] = my_PCA(signal)
    signal = signal';
    signal_mean = mean(signal);
    signal_normalized = signal - signal_mean;
    s = std(signal);
    signal_normalized = signal_normalized ./s;
    
    [CO, SCORE, LATENT] = pca(signal_normalized);
    
    figure;
    scatter(SCORE(:,1), SCORE(:,2), '.')
    xlabel({'','Principle Component 1'})
    ylabel({'','Principle Component 2'})
    
    score = SCORE(:,1:2);
    k = 4;
    [IDX, C, SUMD, D] = kmeans(score,k);
    
    figure;
    hold on
    for i=1:k   
        plot(score(IDX==i,1),score(IDX==i,2), '.', 'MarkerSize',12)
    end
    plot(C(:,1),C(:,2),'kx', 'MarkerSize',15,'LineWidth',3) 
    %legend('Cluster 1','Cluster 2','Cluster 3','Centroids', 'Location','NW')
    title 'K-Means'
    hold off
end
