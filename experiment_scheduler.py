import os
import sys

from run_method import run_experiment

# ['random', 'qbc', 'gradient', 'model_gradient', 'gradient_both_labels',\
#			'gradient_conf', 'gradient_input', 'aligns', 'hamming_max', 'hamming_min']
#990
def scheduler(experiment_i = 0, results_folder = '.'):
	arg_dict = list()
	for noise in [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
		for rand in list(range(100)):
			for method in ['random', 'qbc']:
				arg_dict.append({
				'method':method,
				'random_seed':rand,
				'error_rate':noise,
				'base_antigens_count':5,
				'sample_frac':0.2,
				'data_path':'/cluster/home/roberfra/AbAgAL_main/Data/Processed/ab_ag_binding.tsv',
				'results_dir':f"{results_folder}/noise{noise}/rand{rand}"
				})
	
	exp_tmp = arg_dict[experiment_i]

	os.makedirs(exp_tmp['results_dir'], exist_ok=False)
	run_experiment(**exp_tmp)

if __name__ == '__main__':
	scheduler(experiment_i=int(sys.argv[1]), results_folder=str(sys.argv[2]))