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
	# print(df.iloc[0]['ds'])
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
	output_path = 'output/prophet'
	dpi = 150

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

	# saving files...
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

