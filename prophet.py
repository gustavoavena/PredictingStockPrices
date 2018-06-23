# Python
import pandas as pd
from fbprophet import Prophet, diagnostics
import matplotlib.pyplot as plt
import os
import sys

import dataset_preparation


def test_prophet(train_df, test_df):
	# fpath = os.path.join('dataset', fname)

	# df = pd.read_csv(fpath)
	print(train_df.head())
	print(train_df.tail())
	# print(df.iloc(246))
	
	m = Prophet()
	m.fit(train_df)

	future = m.make_future_dataframe(periods=730)
	# print(future.tail())

	forecast = m.predict(test_df)
	print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

	fig1 = m.plot(forecast)

	fig2 = m.plot_components(forecast)

	plt.show(fig1)
	plt.show(fig2)


def main():

	if(len(sys.argv) > 1):
		fname = sys.argv[1]
		print("Testing prophet with file: {}".format(fname))
	else:
		fname = 'PETR4.SA.csv'

	full_df = dataset_preparation.get_full_dataframe(fname)
	train_df, test_df = dataset_preparation.get_train_test_dataframe(fname)

	test_prophet(train_df, full_df)



if __name__ == '__main__':
	main()

