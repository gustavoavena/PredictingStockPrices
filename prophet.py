# Python
import pandas as pd
from fbprophet import Prophet, diagnostics
from fbprophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt
import os
import sys
from datetime import date

import dataset_preparation


def get_dataset_duration(df):
	# print(df.iloc[0]['ds'])
	first = [int(d) for d in df.iloc[0]['ds'].split('-')]
	last = [int(d) for d in df.iloc[-1]['ds'].split('-')]

	d0 = date(first[0], first[1], first[2])
	d1 = date(last[0], last[1], last[2])
	delta = d1 - d0
	return delta.days




def test_prophet(train_df, test_df):

	print(train_df.head())
	print(train_df.tail())
	
	full_m = Prophet()
	full_m.fit(test_df)


	m = Prophet()
	m.fit(train_df)

	from pprint import pprint


	future = m.make_future_dataframe(periods=730)
	# print(future.tail())



	forecast = m.predict(test_df)
	print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

	fig1 = m.plot(forecast)
	fig1.suptitle('{} prediction model'.format(fname))
	

	plt.show(fig1)



def test_prophet_cross_validation(df, fname, initial_ratio=0.6, period_ratio=0.05, horizon_ratio=0.1, rolling_window=0.1):
	output_path = '../output/prophet'

	print("Performing cross validation on file: ", fname)
	# print(df.head())
	# print(df.tail())
	
	duration = get_dataset_duration(df)

	initial = "{} days".format(int(initial_ratio * duration))
	period = "{} days".format(int(period_ratio * duration))
	horizon = "{} days".format(int(horizon_ratio * duration))

	print("df.shape[0] = {}, duration = {}, initial = {}, period = {}, horizon = {}".format(df.shape[0], duration, initial, period, horizon))

	print("Fitting the model...")
	m = Prophet()
	m.fit(df)

	# df_cv = diagnostics.cross_validation(m, initial='1825 days', period='180 days', horizon = '365 days')
	df_cv = diagnostics.cross_validation(m, initial=initial, period=period, horizon=horizon)
	# print(df_cv)

	df_cv.to_csv(os.path.join(output_path, fname.replace('.csv', '_cross_validation.csv')))

	# print(df_cv)

	# performance metrics
	df_p = diagnostics.performance_metrics(df_cv, rolling_window=rolling_window)
	print(df_p)

	df_p.to_csv(os.path.join(output_path, fname.replace('.csv', '_performance_metrics.csv')))


	fig1 = plot_cross_validation_metric(df_cv, metric='mape', rolling_window=rolling_window)
	fig1.suptitle('{} cross_validation MAPE'.format(fname))
	plt.show(fig1)

	# fig3 = plot_cross_validation_metric(df_cv, metric='mse')
	# fig2.suptitle('{} cross_validation MSE'.format(fname))
	# plt.show(fig3)

def main():

	if(len(sys.argv) > 1):
		fname = sys.argv[1]
		print("Testing prophet with file: {}".format(fname))
	else:
		fname = 'PETR4.SA.csv'

	full_df = dataset_preparation.get_full_dataframe(fname)
	train_df, test_df = dataset_preparation.get_train_test_dataframe(fname)

	# test_prophet(train_df, full_df)
	test_prophet_cross_validation(full_df, fname)



if __name__ == '__main__':
	main()

