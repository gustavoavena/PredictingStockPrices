#!/usr/local/bin/python3

import pandas as pd
from fbprophet import Prophet, diagnostics
from fbprophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt
import os
import sys
from datetime import date

import dataset_preparation

DPI = 150
RESOLUTION = (3840, 2160)


def get_dataset_duration(df):
	first = [int(d) for d in df.iloc[0]['ds'].split('-')]
	last = [int(d) for d in df.iloc[-1]['ds'].split('-')]

	d0 = date(first[0], first[1], first[2])
	d1 = date(last[0], last[1], last[2])
	delta = d1 - d0
	return delta.days


def fit_model_with_prophet(df, fname, future_period=730):
	output_path = 'output/prophet'
	
	print(df.head())
	print(df.tail())

	m = Prophet()
	m.fit(df)

	future = m.make_future_dataframe(periods=future_period)
	print(future.tail())

	forecast = m.predict(future)
	print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

	fig1 = m.plot(forecast)
	fig1.suptitle('{} prediction model'.format(fname))
	
	fig1.figsize = (RESOLUTION[0]/DPI, RESOLUTION[1]/DPI)
	
	figure_path = os.path.join(output_path, fname.replace('.csv', '_model.png'))
	fig1.savefig(figure_path, dpi=DPI)

	# plt.show(fig1)


def prophet_cross_validation(df, fname, initial_ratio=0.6, period_ratio=0.05, horizon_ratio=0.1, rolling_window=0.1):
	"""

	This function performs cross validation with the provided dataset and outputs all metrics and a graph, so the performance can be analyzed.
	It outputs all these files in a folder for a specific set of parameters. For example, running this function with the default parameters will
	create a folder called "cv_0.6_0.05_0.1", which means {cross_validation}_{initial_ratio}_{period_ratio}_{horizon_ratio}.


	**How does Prophet perform cross validation?**

	Prophet performs cross validation using cutoff dates. For example, if your dataset has a range of 10000 days, and initial_ratio = 0.6, period_ratio = 0.05
	and horizon_ratio = 0.1, cross validation we'll be executed in the following way (everything represented in days).

	First cutoff date: last_date - (horizon) = 10000 - (10000*0.1) = 9000.

	So the first cutoff date will be the 9000th day. Prophet will fit the model using all the data prior to this day and will make predictions and calculate
	errors for all days between 9000 and 10000 (cutoff date + horizon).

	After this, it will "skip" backwards a period, in this case, 500 days.

	Second cutoff date: 9000-500 = 8500.

	It will fit the model again, using only data prior to the cutoff date. It will then calculate the prediction and errors for the days in a horizon range
	(in this iteration, from day 8500 to day 9500).

	It will repeat this process until the cutoff date reaches the initial range, which in this case is day 6000.
	It's important to set a significant initial ratio, so the model has enough data to find seasonal patterns and make predictions.


	:param df: dataset in a Pandas dataframe.
	:param fname: original filename to determine the output file names.
	:param initial_ratio: the percentage of the dataset duration to be used as the initial period during cross validation.
	:param period_ratio: the ratio of days that will be skipped between each cross validation iteration.
	:param horizon_ratio: the ratio of days that will be used as horizon between each cross validation iteration.
	:param rolling_window: the ratio of days that will be used to calculate the errors (using rolling means) in the performance_metrics function.
	:return:
	"""
	output_path = 'output/prophet'

	print(fname.replace('.csv', '_{}_{}_{}.csv'.format(initial_ratio, period_ratio, horizon_ratio)))

	print("Performing cross validation on file: ", fname)

	
	duration = get_dataset_duration(df)

	initial = "{} days".format(int(initial_ratio * duration))
	period = "{} days".format(int(period_ratio * duration))
	horizon = "{} days".format(int(horizon_ratio * duration))

	print("df.shape[0] = {}, duration = {}, initial = {}, period = {}, horizon = {}".format(df.shape[0], duration, initial, period, horizon))

	print("Fitting the model...")
	m = Prophet()
	m.fit(df)

	df_cv = diagnostics.cross_validation(m, initial=initial, period=period, horizon=horizon)

	# performance metrics
	df_p = diagnostics.performance_metrics(df_cv, rolling_window=rolling_window)
	print(df_p)

	fig1 = plot_cross_validation_metric(df_cv, metric='mape', rolling_window=rolling_window)
	fig1.suptitle('{} cross_validation MAPE'.format(fname))
	fig1.figsize = (RESOLUTION[0]/DPI, RESOLUTION[1]/DPI)

	# saving files
	suffix = '_{}_{}_{}'.format(initial_ratio, period_ratio, horizon_ratio)
	output_folder = os.path.join(output_path, 'cv' + suffix)

	if(not os.path.exists(output_folder)):
		print('creating output folder: ', output_folder)
		os.mkdir(output_folder)

	df_cv.to_csv(os.path.join(output_folder, fname.replace('.csv', suffix + '_cross_validation.csv')))

	df_p.to_csv(os.path.join(output_folder, fname.replace('.csv', suffix + '_performance_metrics.csv')))

	figure_output = os.path.join(output_folder, fname.replace('.csv', suffix + '_mape.png'))
	fig1.savefig(figure_output, dpi=DPI)

	# plt.show(fig1)



def main():
	if(len(sys.argv) < 2):
		print("Please provide a csv file with data...")

	for fname in sys.argv[1:]:
		print("Processing file: ", fname)
		full_df = dataset_preparation.get_full_dataframe(fname)
		# train_df, test_df = dataset_preparation.get_train_test_dataframe(fname)

		fit_model_with_prophet(full_df, fname)
		# prophet_cross_validation(full_df, fname)


if __name__ == '__main__':
	main()

