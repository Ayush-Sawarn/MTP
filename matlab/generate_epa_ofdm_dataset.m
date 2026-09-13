function generate_epa_ofdm_dataset(pilotCount, outputFile)
%GENERATE_EPA_OFDM_DATASET Generate the EPA OFDM dataset in MATLAB.
%
% Requires LTE Toolbox because it uses lteFadingChannel with DelayProfile
% 'EPA'. This performs sample-level multipath fading with MATLAB's fractional
% delay filters and GMEDS fading model.
%
% Examples (MATLAB Online):
%   generate_epa_ofdm_dataset(24, 'epa7_24pilots_matlab.mat')
%   generate_epa_ofdm_dataset(48, 'epa7_48pilots_matlab.mat')
%
% Saved arrays follow Python-friendly ordering:
%   X_LS    [sample, real_imag, pilot_frequency, pilot_time]
%   H_TRUE  [sample, real_imag, subcarrier, symbol]
%   SNR_DB  [sample, 1]
%
% The paper does not state pilot coordinates, SNR allocation, or MATLAB
% channel-object settings. Those three decisions are made explicitly below
% and saved in METADATA. Adjust them only if the authors' code becomes known.

    arguments
        pilotCount (1,1) double {mustBeMember(pilotCount,[24 48])}
        outputFile (1,:) char = 'epa7_matlab.mat'
    end

    assert(exist('lteFadingChannel','file') == 2, ...
        ['lteFadingChannel was not found. Run this script in MATLAB Online ', ...
         'with LTE Toolbox enabled.']);

    %% Published paper parameters (Table I)
    nSubcarriers = 72;
    nSymbols = 14;
    subcarrierSpacingHz = 15e3;
    carrierFrequencyHz = 2.5e9;
    receiverSpeedKmh = 50;
    cpLength = 16;
    modulation = 'QPSK';
    numSamples = 20000;
    nTrain = 12000;
    nValidation = 4000;
    nTest = 4000;
    assert(nTrain + nValidation + nTest == numSamples);

    %% Derived custom-OFDM quantities
    sampleRateHz = nSubcarriers * subcarrierSpacingHz; % 1.08 MHz
    c = 299792458;
    maximumDopplerHz = (receiverSpeedKmh / 3.6) * carrierFrequencyHz / c;

    %% Unpublished choices: uniform pilot grids and equal SNR allocation.
    % MATLAB's round produces the same intended 6x4 / 12x4 grids used by
    % the Python reproduction. Indices are MATLAB 1-based.
    nPilotTime = 4;
    nPilotFrequency = pilotCount / nPilotTime;
    pilotRows = round(linspace(1, nSubcarriers, nPilotFrequency));
    pilotColumns = round(linspace(1, nSymbols, nPilotTime));
    snrValuesDb = [0 5 10 15 20];
    snrDbPerSample = repmat(snrValuesDb, 1, ceil(numSamples / numel(snrValuesDb)));
    snrDbPerSample = snrDbPerSample(1:numSamples).';
    baseSeed = 2025;

    %% MATLAB LTE Toolbox EPA channel configuration.
    % EPA specifies the 7-path 3GPP delay/power profile. Setting the actual
    % maximum Doppler derived from the paper's carrier/speed is intentional;
    % LTE Toolbox's named EPA5 profile would otherwise use 5 Hz.
    channel = struct;
    channel.DelayProfile = 'EPA';
    channel.NRxAnts = 1;
    channel.DopplerFreq = maximumDopplerHz;
    channel.MIMOCorrelation = 'Low';  % irrelevant for SISO, required by API
    channel.SamplingRate = sampleRateHz;
    channel.InitTime = 0;
    channel.InitPhase = 'Random';
    channel.ModelType = 'GMEDS';
    channel.NTerms = 16;
    channel.NormalizeTxAnts = 'On';
    channel.NormalizePathGains = 'On';

    %% Preallocate single-precision output arrays (~177 MB for 24 pilots).
    X_LS = zeros(numSamples, 2, nPilotFrequency, nPilotTime, 'single');
    H_TRUE = zeros(numSamples, 2, nSubcarriers, nSymbols, 'single');

    % Establish MATLAB's channel-filter delay once. The physical EPA path
    % delays are fractional at 1.08 MHz, so this delay must be compensated
    % before OFDM demodulation.
    channel.Seed = baseSeed;
    [~, channelInfo] = lteFadingChannel(channel, zeros(nSubcarriers + cpLength,1));
    channelFilterDelay = channelInfo.ChannelFilterDelay;

    rng(baseSeed, 'twister');
    fprintf('Generating %d EPA samples (%d pilots) -> %s\n', numSamples, pilotCount, outputFile);
    for sampleIndex = 1:numSamples
        % Independent unit-energy QPSK data/pilot symbols across the grid.
        qpskIndices = randi([0 3], nSubcarriers, nSymbols);
        transmittedGrid = exp(1i * (pi/4 + (pi/2) * qpskIndices));

        % Custom N=72 OFDM modulation with the paper's CP length of 16.
        txWaveform = ofdmModulate(transmittedGrid, nSubcarriers, cpLength);

        % A different deterministic seed gives an independent EPA realization
        % for each sample while preserving full MATLAB reproducibility.
        channel.Seed = baseSeed + sampleIndex - 1;
        paddedTx = [txWaveform; zeros(channelFilterDelay + nSubcarriers + cpLength, 1)];
        rxWithDelay = lteFadingChannel(channel, paddedTx);
        rxWaveform = rxWithDelay(channelFilterDelay + (1:numel(txWaveform)));

        % Perfectly synchronized frequency-domain observation. Hclean is the
        % per-resource-element no-noise channel target used by the paper's
        % y[i,j] = h[i,j]x[i,j] + w[i,j] model.
        receivedGridClean = ofdmDemodulate(rxWaveform, nSubcarriers, cpLength, nSymbols);
        hClean = receivedGridClean ./ transmittedGrid;

        % Add unit-power complex AWGN in the frequency domain at the selected
        % SNR, then form LS estimates only at pilot positions.
        snrDb = snrDbPerSample(sampleIndex);
        noiseSigma = sqrt(10^(-snrDb / 10) / 2);
        receivedGrid = receivedGridClean + noiseSigma * ...
            (randn(nSubcarriers,nSymbols) + 1i*randn(nSubcarriers,nSymbols));
        hLsPilots = receivedGrid(pilotRows,pilotColumns) ./ ...
            transmittedGrid(pilotRows,pilotColumns);

        X_LS(sampleIndex,1,:,:) = reshape(single(real(hLsPilots)), [1 1 nPilotFrequency nPilotTime]);
        X_LS(sampleIndex,2,:,:) = reshape(single(imag(hLsPilots)), [1 1 nPilotFrequency nPilotTime]);
        H_TRUE(sampleIndex,1,:,:) = reshape(single(real(hClean)), [1 1 nSubcarriers nSymbols]);
        H_TRUE(sampleIndex,2,:,:) = reshape(single(imag(hClean)), [1 1 nSubcarriers nSymbols]);

        if mod(sampleIndex, 500) == 0 || sampleIndex == numSamples
            fprintf('  %d / %d samples complete\n', sampleIndex, numSamples);
        end
    end

    %% Save data and all published/unpublished parameters together.
    split = struct('train', 1:nTrain, ...
                   'validation', nTrain + (1:nValidation), ...
                   'test', nTrain + nValidation + (1:nTest));
    metadata = struct;
    metadata.paperParameters = struct( ...
        'channelModel','3GPP EPA, 7 paths', ...
        'carrierFrequencyHz',carrierFrequencyHz, ...
        'receiverSpeedKmh',receiverSpeedKmh, ...
        'subcarrierSpacingHz',subcarrierSpacingHz, ...
        'nSubcarriers',nSubcarriers, ...
        'nSymbols',nSymbols, ...
        'modulation',modulation, ...
        'cpLength',cpLength, ...
        'totalSamples',numSamples, ...
        'splitSizes',[nTrain nValidation nTest]);
    metadata.derivedParameters = struct( ...
        'sampleRateHz',sampleRateHz, ...
        'maximumDopplerHz',maximumDopplerHz, ...
        'channelFilterDelaySamples',channelFilterDelay, ...
        'modelType','GMEDS', ...
        'nTerms',16);
    metadata.pilotRowsMatlabOneBased = pilotRows;
    metadata.pilotColumnsMatlabOneBased = pilotColumns;
    metadata.snrValuesDb = snrValuesDb;
    metadata.baseSeed = baseSeed;
    metadata.assumptions = { ...
        'Pilot coordinates are not reported in the paper; a uniform grid is used.', ...
        'The training SNR allocation is not reported; samples are equally allocated to 0,5,10,15,20 dB.', ...
        'The paper does not report MATLAB random seeds or LTE fading-object options; GMEDS, 16 terms, and independent seeds are used.'};
    metadata.arrayLayout = struct( ...
        'X_LS','[sample, real_imag, pilot_frequency, pilot_time]', ...
        'H_TRUE','[sample, real_imag, subcarrier, symbol]', ...
        'SNR_DB','[sample, 1]');

    SNR_DB = single(snrDbPerSample);
    save(outputFile, 'X_LS', 'H_TRUE', 'SNR_DB', 'split', 'metadata', '-v7.3');
    fprintf('Saved %s\n', outputFile);
end


function waveform = ofdmModulate(resourceGrid, nFft, cpLength)
% Unit-power IFFT convention: fft(ofdmSymbol)/sqrt(N) recovers the grid.
    timeSymbols = ifft(resourceGrid, nFft, 1) * sqrt(nFft);
    withCp = [timeSymbols(end-cpLength+1:end,:); timeSymbols];
    waveform = withCp(:);
end


function resourceGrid = ofdmDemodulate(waveform, nFft, cpLength, nSymbols)
    symbolLength = nFft + cpLength;
    assert(numel(waveform) == nSymbols * symbolLength, 'Unexpected waveform length after delay compensation.');
    withCp = reshape(waveform, symbolLength, nSymbols);
    timeSymbols = withCp(cpLength+1:end,:);
    resourceGrid = fft(timeSymbols, nFft, 1) / sqrt(nFft);
end
