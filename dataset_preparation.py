import os
import sys

import pandas as pd

index_by_reference_price = {
	'open': 1,
	'high': 2,
	'low': 3,
	'close': 4,
	'adj_close': 5 
}


def get_full_dataframe(fname, reference_price='open'):
	
	fpath = os.path.join('dataset/originals', fname)
	# output_path = os.path.join('dataset', fname.replace('.csv', '_clean.csv'))

	y_index = index_by_reference_price[reference_price]

	f = open(fpath, 'r')

	f.readline()


	d = {'ds':[], 'y': []}


	for line in f:
		split = line.split(',')
		if(split[y_index] == 'null'):
			continue
		d['ds'].append(split[0])
		d['y'].append(split[y_index])
		
	f.close()

	return pd.DataFrame(d)




def get_train_test_dataframe(fname, reference_price='open'):

	df = get_full_dataframe(fname, reference_price)

	rows = df.shape[0]

	# print(rows)
	train_cut = int(rows * 0.75)

	


	return df[:train_cut], df[train_cut:]

	




def main():

	if(len(sys.argv) > 1):
		fname = sys.argv[1]
	else:
		fname = 'PETR4.SA.csv'
	
	export_clean_dataset(fname, dataset='test')

if __name__ == '__main__':
	main()