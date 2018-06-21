# Python
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
import os
import sys

from dataset_preparation import get_clean_dataframe


def test_prophet(fname):
	fpath = os.path.join('dataset', fname)

	df = pd.read_csv(fpath)
	print(df.head())
	
	m = Prophet()
	m.fit(df)

	future = m.make_future_dataframe(periods=730)
	print(future.tail())

	forecast = m.predict(future)
	print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

	fig1 = m.plot(forecast)


	plt.show(fig1)


def main():

	if(len(sys.argv) > 1):
		fname = sys.argv[1]
	else:
		fname = 'PETR4.SA.csv'

	get_clean_dataframe(fname)
	fname = fname.replace('.csv', '_clean.csv')
	test_prophet(fname)



if __name__ == '__main__':
	main()

