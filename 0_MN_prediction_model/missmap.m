% Read the data from the Excel file
variables = {'LL (%)', 'PI (%)', 'LI', '(qt-svo)/s¢vo', '(qt-u2)/s¢vo', '(u2-u0)/s¢vo', 'Bq', 'su(mob)/s¢v0'};
original_data = readtable('original_data.xlsx');

% Rename columns to valid MATLAB variable names
original_data.Properties.VariableNames = matlab.lang.makeValidName(original_data.Properties.VariableNames);

% Identify missing data
missing_data = ismissing(original_data);

% Plot missing data as a binary heatmap
figure;
imagesc(missing_data);
colormap(flipud(gray)); % Optional, depending on how you'd like to display missing data
colorbar; % Adds a color bar to indicate missing (1) and non-missing (0) values

% Add axis labels
xticks(1:size(missing_data, 2));
yticks(1:size(missing_data, 1));
xticklabels(original_data.Properties.VariableNames);
yticklabels(1:size(missing_data, 1));

% Rotate x-axis labels for better visibility
xtickangle(45);

% Set axis labels and title
xlabel('Variables');
ylabel('Observations');
title('Missing Data Matrix');

% Save the plot
output_folder = fullfile('results');
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end
output_path = fullfile(output_folder, 'missing_map.png');
saveas(gcf, output_path);

% Display the plot
disp('Missing data matrix plot saved as missing_map.png');

