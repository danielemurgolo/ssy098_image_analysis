function [normalizedData] = normalize_data(data)
% Function Name: normalize_data
%
% Description: This function normalizes input data using mean and standard
%              deviation.
% 
% Inputs:
%   - data: Input data to be normalized
%
% Outputs:
%   - normalizedData: Normalized data
%
% Example Usage:
%   >> normalizedData = normalize_data([1, 2, 3, 4, 5])
%
% Author: Daniele Murgolo
% Date: March 6th, 2023
normalizedData  = (data  - mean(data))./std(data);
end